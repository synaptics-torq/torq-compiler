import pytest

import json
from pathlib import Path

import onnx
from onnx import shape_inference

from torq.testing.comparison import compare_test_results
from torq.testing.onnx import generate_onnx_layers_from_file, has_bf16_matmul, has_bf16_einsum
from torq.testing.iree import llvmcpu_reference_results
from torq.testing.iree import  list_files

"""ONNX model layer/full-model test harness.

Any ONNX model under ``tests/testdata/dev_ops`` or ``tests/testdata/onnx_models``
can be tested layer-by-layer or as a full model.

Basic usage:

    # List all generated cases
    pytest tests/test_onnx_model.py -v -s --collect-only

    # Run the full model
    pytest tests/test_onnx_model.py -v -s [filename.stem]_full_model

    # Run a single extracted layer
    pytest tests/test_onnx_model.py -v -s [filename.stem]_layer_[layername]
    # e.g. pytest tests/test_onnx_model.py -v -s mbv2.quant_layer_DequantizeLinear_306

Quantization:

    This test already supports the shared ONNX quantization flags.  To run the
    same cases under int8 quantization, pass ``--quantize`` (and usually
    ``--full-integer`` so graph I/O remain int8):

        pytest tests/test_onnx_model.py -v -s --quantize --full-integer --quant-format=qdq

    With ``--quant-format=qoperator`` the model is rewritten to use native
    quantized ops such as ``QLinearConv`` instead of QDQ nodes.
    
"""

@pytest.fixture
def case_config(request, chip_config):

    example_tc = [
        # unnecessary to run, only for executor discovery test
        "example_gen_config",
    ]
    if any(s in request.node.nodeid for s in example_tc):
        pytest.xfail("Skipped Example Cases")
    
    return {
         "onnx_model": "onnx_layer_model",
         "mlir_model_file": "onnx_mlir_model_file",
         "input_data": "tweaked_random_input_data",
    }


def pytest_generate_tests(metafunc):
    files = list_files("dev_ops", ".onnx", False) + list_files("onnx_models", ".onnx", False)

    if not files:
        return

    cases = []
    node_groups = [
            ['Conv', 'Clip'],
        ]

    for f in files:
        cases += generate_onnx_layers_from_file(metafunc.config.cache, f, node_groups)

    metafunc.parametrize("onnx_layer_model", cases, indirect=True, ids=[c.name for c in cases])

@pytest.fixture
def reference_results(request, onnx_layer_model):
    """Select reference: numpy for bf16 MatMul and Einsum, llvmcpu otherwise."""

    if has_bf16_einsum(onnx_layer_model.data) or has_bf16_matmul(onnx_layer_model.data):
        numpy_reference_results = request.getfixturevalue("numpy_reference_results")
        return numpy_reference_results

    llvmcpu_reference_results = request.getfixturevalue("llvmcpu_reference_results")
    return llvmcpu_reference_results

# Not ready for that (Nan differs)
# @pytest.mark.fpga_ci
@pytest.mark.ci
def test_onnx_model_llvmcpu_torq(request, reference_results, torq_results, case_config, onnx_layer_model):
    compare_test_results(request, torq_results, reference_results, case_config)
