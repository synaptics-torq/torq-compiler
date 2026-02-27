"""
Layer-by-layer testing of TFLite models.

This test module extracts individual layers from TFLite models and tests them
against reference implementations (LLVM-CPU) and compares with Torq backend.

Approach:
1. Extract individual layers directly from TFLite model using flatbuffer manipulation
2. Quantization parameters are preserved exactly from the original model
3. Test each layer independently (unsupported ops only affect that layer)

The extraction preserves int8 quantization from the original model, ensuring
layers remain quantized with appropriate scale/zero_point values.

Usage:
    # See all test cases:
    pytest extras/tests/test_tflite_model.py -v -s --collect-only

    # Run full model test:
    pytest extras/tests/test_tflite_model.py -v -s -k "full_model"

    # Run specific layer:
    pytest extras/tests/test_tflite_model.py -v -s -k "layer_CONV_2D"
    
    # Limit number of layers (set MAX_LAYERS environment variable):
    MAX_LAYERS=5 pytest extras/tests/test_tflite_model.py -v -s --collect-only
    
    # Force re-extraction of layers (clear cache):
    FORCE_EXTRACT=1 pytest extras/tests/test_tflite_model.py -v -s --collect-only
"""

import pytest
import subprocess
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Any, List

# Import the direct TFLite layer extractor (preserves quantization)
from torq.testing.tflite_layer_extractor import extract_all_layers

from torq.testing.comparison import compare_test_results
from torq.testing.iree import llvmcpu_reference_results, _find_iree_tool
from torq.testing.versioned_fixtures import (
    versioned_generated_file_fixture,
    versioned_static_file_fixture,
    VersionedUncachedData,
)

# Configuration: limit number of layers to test (0 = no limit)
MAX_LAYERS = int(os.environ.get('MAX_LAYERS', '0'))
FORCE_EXTRACT = os.environ.get('FORCE_EXTRACT', '0') == '1'

# Global cache directory
_MLIR_CACHE_DIR = None


@dataclass
class TFLiteLayerCase:
    """Represents a test case for a TFLite layer or full model."""
    name: str
    data: Any


# ============================================================================
# TOSA/MLIR Conversion
# ============================================================================

def convert_tflite_to_tosa(tflite_path: Path, output_path: Path) -> bool:
    """Convert TFLite to TOSA using iree-import-tflite."""
    try:
        result = subprocess.run(
            ["iree-import-tflite", str(tflite_path), "-o", str(output_path)],
            capture_output=True, text=True, timeout=60
        )
        return result.returncode == 0
    except Exception as e:
        print(f"  TOSA conversion failed: {e}")
        return False


# ============================================================================
# Test Case Generation
# ============================================================================

# Cache for generated test cases (avoid regenerating on each pytest_generate_tests call)
_CASES_CACHE = {}

def get_mlir_cache_dir() -> Path:
    """Get or create the global MLIR cache directory."""
    global _MLIR_CACHE_DIR
    if _MLIR_CACHE_DIR is None:
        _MLIR_CACHE_DIR = Path(__file__).parent / "testdata" / "tflite_models" / ".mlir_cache"
        _MLIR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return _MLIR_CACHE_DIR


def generate_tflite_layer_cases(tflite_path: Path, max_layers: int = 0) -> List[TFLiteLayerCase]:
    """
    Generate test cases for each layer in a TFLite model.
    
    Uses the direct TFLite layer extractor (Option 2) which preserves 
    quantization from the original model.
    
    Args:
        tflite_path: Path to TFLite model file
        max_layers: Maximum number of layers to extract (0 = no limit)
        
    Returns:
        List of TFLiteLayerCase objects
    """
    import json as json_mod
    
    cases = []
    
    model_stem = tflite_path.stem
    cache_dir = get_mlir_cache_dir()
    layers_dir = cache_dir / f"{model_stem}_layers"
    layers_dir.mkdir(parents=True, exist_ok=True)
    
    # Check for cached cases file (fast path - no TF import needed)
    cases_cache_file = layers_dir / "_cases_cache.json"
    if cases_cache_file.exists() and not FORCE_EXTRACT:
        try:
            with open(cases_cache_file, 'r') as f:
                cached_cases = json_mod.load(f)
            # Reconstruct TFLiteLayerCase objects
            result = []
            for c in cached_cases:
                if max_layers > 0 and len(result) >= max_layers + 1:  # +1 for full model
                    break
                result.append(TFLiteLayerCase(name=c['name'], data=c['data']))
            print(f"Loaded {len(result)} cached cases for {model_stem} (use FORCE_EXTRACT=1 to regenerate)")
            return result
        except Exception as e:
            print(f"Cache load failed for {model_stem}: {e}")
    
    # Use the new direct layer extractor that preserves quantization
    print(f"Processing {tflite_path.name} with quantization-preserving extractor...")
    
    try:
        extraction_results = extract_all_layers(
            str(tflite_path),
            str(layers_dir),
            max_layers=max_layers,
            force=FORCE_EXTRACT,
        )
    except Exception as e:
        print(f"ERROR: Failed to extract layers from {tflite_path.name}: {e}")
        # Return only the full-model case so test collection doesn't abort
        cases.append(TFLiteLayerCase(
            name=f"{model_stem}_full_model",
            data={'model_path': str(tflite_path), 'is_layer': False}
        ))
        return cases
    
    for result in extraction_results:
        op_name = result['op_name']
        op_index = result['layer_index']
        layer_name = f"{model_stem}_layer_{op_name}_{op_index}"
        
        if not result['success']:
            # Create case anyway but mark as failed
            cases.append(TFLiteLayerCase(name=layer_name, data={
                'model_path': str(tflite_path),
                'op_name': op_name,
                'op_index': op_index,
                'is_layer': True,
                'extraction_failed': True,
                'is_quantized': result.get('is_quantized', False),
            }))
            continue
        
        layer_tflite = Path(result['layer_file'])
        layer_tosa = layer_tflite.with_suffix('.tosa')
        
        # Convert to TOSA
        if layer_tflite.exists() and (FORCE_EXTRACT or not layer_tosa.exists()):
            print(f"  Converting {op_name} to TOSA/MLIR...")
            convert_tflite_to_tosa(layer_tflite, layer_tosa)
        
        # Also save text-format MLIR alongside the TOSA bytecode
        layer_mlir = layer_tflite.with_suffix('.mlir')
        if layer_tosa.exists() and (FORCE_EXTRACT or not layer_mlir.exists()):
            try:
                iree_opt = _find_iree_tool('IREE_OPT', 'iree-opt')
                subprocess.run(
                    [iree_opt, str(layer_tosa), "-o", str(layer_mlir)],
                    capture_output=True, text=True, timeout=60
                )
            except (FileNotFoundError, subprocess.TimeoutExpired) as e:
                print(f"  Could not save text MLIR for {op_name}: {e}")
        
        cases.append(TFLiteLayerCase(name=layer_name, data={
            'model_path': str(tflite_path),
            'layer_tflite_path': str(layer_tflite),
            'tosa_path': str(layer_tosa) if layer_tosa.exists() else None,
            'op_name': op_name,
            'op_index': op_index,
            'is_layer': True,
            'is_tosa_layer': layer_tosa.exists(),
            'is_quantized': result.get('is_quantized', False),
            'inputs': result.get('inputs', []),
            'outputs': result.get('outputs', []),
        }))
    
    # Add full model case
    cases.append(TFLiteLayerCase(name=f"{model_stem}_full_model", data={
        'model_path': str(tflite_path),
        'is_layer': False,
    }))
    
    # Save cases cache for fast loading next time
    try:
        with open(cases_cache_file, 'w') as f:
            json_mod.dump([{'name': c.name, 'data': c.data} for c in cases], f)
    except Exception:
        pass  # Ignore cache write errors
    
    print(f"  Generated {len(cases)} test cases ({sum(1 for r in extraction_results if r['success'])} layers + 1 full model)")
    return cases


def list_tflite_files(directory: str = "tflite_models") -> List[Path]:
    """List all TFLite files in the given directory."""
    tflite_dir = Path(__file__).parent / "testdata" / directory
    if not tflite_dir.exists():
        return []
    return list(tflite_dir.glob("*.tflite"))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def case_config(request, chip_config):
    """Configure test case settings."""
    return {
        "tflite_model": "tflite_layer_model",
        "mlir_model_file": "tflite_mlir_model_file",
        "input_data": "tweaked_random_input_data",
    }


@pytest.fixture
def tflite_layer_model(request):
    """Fixture that provides the TFLite model for the current test case."""
    case = request.param
    version = "tflite_layer_model_" + case.name
    return VersionedUncachedData(data=case.data, version=version)


@pytest.fixture
def tflite_model_path(tflite_layer_model):
    """Get the TFLite model path."""
    data = tflite_layer_model.data if isinstance(tflite_layer_model, VersionedUncachedData) else tflite_layer_model
    if isinstance(data, dict):
        # For layers, use the extracted layer TFLite if available
        layer_path = data.get('layer_tflite_path')
        if layer_path and Path(layer_path).exists():
            return layer_path
        return data.get('model_path')
    return str(data)


@versioned_static_file_fixture
def tflite_model_file(request, tflite_model_path):
    """Provide the TFLite model file path."""
    return Path(tflite_model_path)


@versioned_generated_file_fixture("mlir")
def tflite_mlir_model_file(request, versioned_file, tflite_model_file):
    """Convert TFLite model to MLIR format (text)."""
    mlir_output = str(versioned_file)
    tosa_output = mlir_output.replace('.mlir', '.tosa')
    
    # Step 1: iree-import-tflite -> TOSA bytecode
    result = subprocess.run(
        ["iree-import-tflite", str(tflite_model_file), "-o", tosa_output],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        pytest.skip(f"Failed to convert TFLite to TOSA: {result.stderr}")
    
    # Step 2: iree-opt -> text MLIR
    try:
        iree_opt = _find_iree_tool('IREE_OPT', 'iree-opt')
    except FileNotFoundError:
        pytest.skip("iree-opt not found, could not convert to text MLIR")
    
    result = subprocess.run(
        [iree_opt, tosa_output, "-o", mlir_output],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        pytest.skip(f"iree-opt failed: {result.stderr}")


# ============================================================================
# Test Generation
# ============================================================================

def pytest_generate_tests(metafunc):
    """Generate test cases for all TFLite models."""
    global _CASES_CACHE
    
    if 'tflite_layer_model' not in metafunc.fixturenames:
        return
    
    # Use cached cases if available (pytest_generate_tests is called multiple times)
    cache_key = f"tflite_cases_{MAX_LAYERS}"
    if cache_key in _CASES_CACHE:
        cases = _CASES_CACHE[cache_key]
    else:
        files = list_tflite_files("tflite_models")
        if not files:
            print("No TFLite files found in testdata/tflite_models")
            return
        
        cases = []
        for f in files:
            cases.extend(generate_tflite_layer_cases(f, max_layers=MAX_LAYERS))
        
        _CASES_CACHE[cache_key] = cases
    
    if cases:
        metafunc.parametrize(
            "tflite_layer_model",
            cases,
            indirect=True,
            ids=[c.name for c in cases]
        )


# ============================================================================
# Tests
# ============================================================================

def test_tflite_model_llvmcpu_torq(
    request,
    llvmcpu_reference_results,
    torq_results,
    case_config,
    tflite_layer_model,  # needed for pytest_generate_tests parametrization
):
    """Compare TFLite model results between LLVM-CPU and Torq backends."""
    compare_test_results(request, torq_results, llvmcpu_reference_results, case_config)


# ============================================================================
# CLI Utility
# ============================================================================

if __name__ == "__main__":
    """Print info about TFLite models when run directly."""
    files = list_tflite_files("tflite_models")
    for f in files:
        interpreter = tf.lite.Interpreter(model_path=str(f))
        interpreter.allocate_tensors()
        
        print(f"\n{'='*60}")
        print(f"Model: {f}")
        print(f"Inputs: {len(interpreter.get_input_details())}")
        print(f"Outputs: {len(interpreter.get_output_details())}")
        
        try:
            ops = interpreter._get_ops_details()
            print(f"Operators: {len(ops)}")
            op_types = set(op.get('op_name', 'UNKNOWN') for op in ops)
            print(f"Op types: {sorted(op_types)}")
        except:
            pass
