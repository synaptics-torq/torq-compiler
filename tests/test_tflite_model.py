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
    pytest extras/tests/test_tflite_model.py --collect-only

    # Run full model test:
    pytest extras/tests/test_tflite_model.py -v -s -m "full_model"

    # Run specific layer:
    pytest extras/tests/test_tflite_model.py -v -s -k "layer_CONV_2D"
    
    # Limit number of layers:
    pytest extras/tests/test_tflite_model.py -v -s --collect-only --tflite-layer-test-max-layers=5 
    
    # Force re-extraction of layers (clear cache):
    pytest extras/tests/test_tflite_model.py -v -s --collect-only --tflite-layer-test-force-extract
"""

import pytest
from pathlib import Path
from typing import List

from torq.testing.comparison import compare_test_results
from torq.testing.tflite_layer_tests import generate_parametrized_tests


# ============================================================================
# Test Case Generation
# ============================================================================

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
def case_config(request):
    """Configure test case settings."""
    return {
        "tflite_model_file": "tflite_model_path",
        "mlir_model_file": "tflite_mlir_model_file",
        "input_data": "tweaked_random_input_data",
    }

# ============================================================================
# Test Generation
# ============================================================================

def list_tflite_files():
    """List all TFLite files in the testdata/tflite_models directory."""

    tflite_dir = Path(__file__).parent / "testdata" / "tflite_models"
    if not tflite_dir.exists():
        print("No tflite_models directory found")
        return []
    return list(tflite_dir.glob("*.tflite"))


def pytest_generate_tests(metafunc):
    """Generate test cases for all TFLite models in the models directory."""
    
    files = list_tflite_files()

    if not files:
        print("No TFLite files found in testdata/tflite_models")
        return
    
    generate_parametrized_tests(metafunc, [f.stem for f in files], files)


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
    
    files = list_tflite_files()

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
