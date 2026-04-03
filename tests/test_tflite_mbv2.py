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
import numpy as np
from pathlib import Path
from PIL import Image

from torq.testing.comparison import compare_test_results
from torq.testing.hf import get_hf_dataset_file, get_hf_model_file
from torq.testing.versioned_fixtures import (
    versioned_cached_data_fixture
)

from torq.testing.tflite_layer_tests import generate_parametrized_tests, TFLiteLayerCase


def _dequantize_mbv2_output(output_data):
    return (output_data.astype(np.float32) + 128.0) / 256.0


def _compare_full_model_pair(left_results, right_results, left_name, right_name):
    left_output = np.squeeze(left_results.data[0])
    right_output = np.squeeze(right_results.data[0])

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
    """Download MobileNetV2 int8 TFLite model from HuggingFace."""
    return Path(get_hf_model_file(cache, "Synaptics/MobileNetV2", "MobileNetV2_int8.tflite"))


# ============================================================================
# Fixtures
# ============================================================================

def mbv2_compile_options():
    torq_compiler_options = ["--torq-convert-dtypes", "--torq-disable-host"]
    
    # tile-and-fuse is less optimal because of the transpose ops which are tiled in incompatible ways
    # and cannot be folded.
    torq_compiler_options += ["--torq-enable-torq-hl-tiling"]

    return torq_compiler_options


@pytest.fixture
def case_config(request, runtime_hw_type, chip_config):
    """Configure test case settings."""

    # xfail non-SL2610 chips on aws_fpga (known mismatch)
    if chip_config.data.get("target", "SL2610") != "SL2610" and runtime_hw_type.data == "aws_fpga":
        pytest.xfail(f"Known failure for chip {chip_config.data.get('chip_name', 'unknown')} on aws_fpga") # Mismatched elements: 1 / 1000 (0.1%)

    return {
        "full_model_input_data": "mbv2_input_data",
        "tflite_model_file": "tflite_model_path",
        "mlir_model_file": "tflite_mlir_model_file",
        "input_data": "tflite_layer_inputs",
        "torq_compiler_options": mbv2_compile_options()
    }


@versioned_cached_data_fixture
def mbv2_input_data(request):        
    
    cache = request.getfixturevalue("cache")

    image_path = get_hf_dataset_file(cache, "Synaptics/TorqCompilerTestImages", "space_shuttle_224x224.jpg")

    with Image.open(image_path) as image:
        image = image.convert("RGB").resize((224, 224))
        image_array = np.array(image)

    # pre-processing valid for "i8" and "si8"
    input_tensor = (image_array.astype(np.float32) - 127).astype(np.int8)

    # pre-processing valid for "ui8"
    # input_tensor = image_array.astype(np.uint8)    

    return [np.expand_dims(input_tensor, axis=0)]


# ============================================================================
# Test Generation
# ============================================================================

def pytest_generate_tests(metafunc):
    """Generate test cases for MobileNetV2 model."""

    model_path = download_mobilenetv2_model(metafunc.config.cache)

    generate_parametrized_tests(metafunc, ["mbv2"], [model_path])
    

# ============================================================================
# Tests
# ============================================================================

def _compare_results(request, left_results, right_results, left_name, right_name,
                     case_config, tflite_layer_model):
    """Dispatch to full-model or layer comparison."""
    if tflite_layer_model.data.is_layer:        
        compare_test_results(request, right_results, left_results, case_config)
    else:
        _compare_full_model_pair(left_results, right_results, left_name, right_name)
    

def test_mbv2_llvmcpu_torq(
    request,
    llvmcpu_reference_results,
    torq_results,
    case_config,
    tflite_layer_model,
):
    """Compare MobileNetV2 results between LLVM-CPU and Torq backends."""
    _compare_results(request, llvmcpu_reference_results, torq_results,
                     "LLVMCPU", "TORQ", case_config, tflite_layer_model)


@pytest.mark.ci
@pytest.mark.fpga_ci
def test_mbv2_tflite_torq(
    request,
    tflite_reference_results,
    torq_results,
    case_config,
    tflite_layer_model
):
    """Compare MobileNetV2 results between TFLite and Torq backends."""
    _compare_results(request, tflite_reference_results, torq_results,
                     "TFLite", "TORQ", case_config, tflite_layer_model)


def test_mbv2_llvmcpu_tflite(
    request,
    tflite_reference_results,
    llvmcpu_reference_results,
    case_config,
    tflite_layer_model,
):
    """Compare MobileNetV2 results between LLVM-CPU and TFLite backends."""
    _compare_results(request, llvmcpu_reference_results, tflite_reference_results,
                     "LLVMCPU", "TFLite", case_config, tflite_layer_model)
