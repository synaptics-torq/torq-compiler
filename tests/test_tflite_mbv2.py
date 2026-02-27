"""
End-to-end and layer-by-layer testing of MobileNetV2 (int8 quantized).

Downloads the MobileNetV2_int8.tflite model from HuggingFace (Synaptics/MobileNetV2)
and tests it both as a full model and layer-by-layer, reusing the infrastructure
from tests/test_tflite_model.py.

Usage:
    # See all test cases:
    pytest tests/test_mbv2_e2e.py -v -s --collect-only

    # Run full model test only:
    pytest tests/test_mbv2_e2e.py -v -s -k "full_model"

    # Run specific layer type:
    pytest tests/test_mbv2_e2e.py -v -s -k "layer_CONV_2D"

    # Force re-extraction of layers (clear cache):
    FORCE_EXTRACT=1 pytest tests/test_mbv2_e2e.py -v -s --collect-only
"""

import pytest
import subprocess
import os
import sys
import shutil
import numpy as np
from pathlib import Path
from PIL import Image

from torq.testing.comparison import compare_test_results
from torq.testing.hf import get_hf_dataset_file, get_hf_model_file
from torq.testing.iree import _find_iree_tool
from torq.testing.versioned_fixtures import (
    versioned_generated_file_fixture,
    versioned_static_file_fixture,
    VersionedUncachedData,
)

# Ensure tests/ directory is on the import path
TESTS_DIR = Path(__file__).resolve().parent
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

from test_tflite_model import generate_tflite_layer_cases, TFLiteLayerCase

# Directory where test_tflite_model.py looks for tflite files
TFLITE_MODELS_DIR = Path(__file__).resolve().parent / "testdata" / "tflite_models"

# Configuration
MAX_LAYERS = int(os.environ.get('MAX_LAYERS', '0'))


def _fixture_data(value):
    return value.data if hasattr(value, "data") else value


def _is_full_model_case(tflite_layer_model):
    data = _fixture_data(tflite_layer_model)
    return isinstance(data, dict) and not data.get("is_layer", True)


def _dequantize_mbv2_output(output_data):
    return (output_data.astype(np.float32) + 128.0) / 256.0


def _compare_full_model_pair(left_results, right_results, left_name, right_name):
    left_output = np.squeeze(_fixture_data(left_results)[0])
    right_output = np.squeeze(_fixture_data(right_results)[0])

    left_output_deq = _dequantize_mbv2_output(left_output)
    right_output_deq = _dequantize_mbv2_output(right_output)

    left_class = int(np.argmax(left_output_deq))
    right_class = int(np.argmax(right_output_deq))

    print(f"{left_name} predicted class: {left_class}")
    print(f"{right_name} predicted class: {right_class}")

    assert left_class == right_class, f"{left_name} class {left_class} != {right_name} class {right_class}"
    assert right_class == 812, f"Expected space shuttle class 812, got {right_class}"

    np.testing.assert_allclose(left_output_deq, right_output_deq, rtol=0.1, atol=0.1)


# ============================================================================
# Model Download
# ============================================================================

def download_mobilenetv2_model(cache):
    """
    Download MobileNetV2 int8 TFLite model from HuggingFace and copy to the
    tflite_models directory so it can be used by the layer extractor.
    """
    TFLITE_MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = Path(get_hf_model_file(cache, "Synaptics/MobileNetV2", "MobileNetV2_int8.tflite"))
    dest = TFLITE_MODELS_DIR / model_path.name
    if not dest.exists():
        shutil.copy2(model_path, dest)
    return dest


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def case_config(request):
    """Configure test case settings."""
    return {
        "tflite_model": "tflite_layer_model",
        "mlir_model_file": "tflite_mlir_model_file",
        "input_data": "mbv2_input_data",
    }


@pytest.fixture
def mbv2_input_data(request, tflite_layer_model, tweaked_random_input_data, mlir_io_spec):
    if not _is_full_model_case(tflite_layer_model):
        return tweaked_random_input_data

    io_spec = _fixture_data(mlir_io_spec)

    if not io_spec.inputs:
        return tweaked_random_input_data

    input_spec = io_spec.inputs[0]
    if len(input_spec.shape) != 4:
        return tweaked_random_input_data

    cache = request.getfixturevalue("cache")
    image_path = get_hf_dataset_file(cache, "Synaptics/TorqCompilerTestImages", "space_shuttle_224x224.jpg")

    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((224, 224))
        image_array = np.array(image)

    if input_spec.fmt in ("i8", "si8"):
        input_tensor = (image_array.astype(np.float32) - 127).astype(np.int8)
    elif input_spec.fmt == "ui8":
        input_tensor = image_array.astype(np.uint8)
    else:
        return tweaked_random_input_data

    return VersionedUncachedData(
        data=[np.expand_dims(input_tensor, axis=0)],
        version=f"mbv2_full_model_input_{input_spec.fmt}_{'x'.join(map(str, input_spec.shape))}",
    )


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

_CASES_CACHE = {}


def _is_full_model_only(metafunc):
    """Check if -k filter targets only full model tests."""
    keyword_expr = metafunc.config.option.keyword
    if not keyword_expr:
        return False
    return 'full' in keyword_expr.lower()


def pytest_generate_tests(metafunc):
    """Generate test cases for MobileNetV2 model."""
    if 'tflite_layer_model' not in metafunc.fixturenames:
        return

    full_only = _is_full_model_only(metafunc)
    cache_key = f"mbv2_e2e_cases_{MAX_LAYERS}_full={full_only}"
    if cache_key in _CASES_CACHE:
        cases = _CASES_CACHE[cache_key]
    else:
        cache = metafunc.config.cache
        model_path = download_mobilenetv2_model(cache)

        if full_only:
            cases = [
                TFLiteLayerCase(
                    name=f"{model_path.stem}_full_model",
                    data={'model_path': str(model_path), 'is_layer': False}
                )
            ]
        else:
            cases = generate_tflite_layer_cases(model_path, max_layers=MAX_LAYERS)

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

def test_mbv2_llvmcpu_torq(
    request,
    llvmcpu_reference_results,
    torq_results,
    case_config,
    tflite_layer_model,
):
    """Compare MobileNetV2 results between LLVM-CPU and Torq backends."""
    if _is_full_model_case(tflite_layer_model):
        _compare_full_model_pair(llvmcpu_reference_results, torq_results, "LLVMCPU", "TORQ")
        return

    compare_test_results(request, torq_results, llvmcpu_reference_results, case_config)


@pytest.mark.ci
def test_mbv2_tflite_torq(
    request,
    tflite_reference_results,
    torq_results,
    case_config,
    tflite_layer_model,
):
    """Compare MobileNetV2 results between TFLite and Torq backends."""
    if _is_full_model_case(tflite_layer_model):
        _compare_full_model_pair(tflite_reference_results, torq_results, "TFLite", "TORQ")
        return

    compare_test_results(request, torq_results, tflite_reference_results, case_config)


def test_mbv2_llvmcpu_tflite(
    request,
    tflite_reference_results,
    llvmcpu_reference_results,
    case_config,
    tflite_layer_model,
):
    """Compare MobileNetV2 results between LLVM-CPU and TFLite backends."""
    if _is_full_model_case(tflite_layer_model):
        _compare_full_model_pair(llvmcpu_reference_results, tflite_reference_results, "LLVMCPU", "TFLite")
        return

    compare_test_results(request, tflite_reference_results, llvmcpu_reference_results, case_config)
