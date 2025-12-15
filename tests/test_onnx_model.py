import pytest

import json
from pathlib import Path

import onnx
from onnx import shape_inference

from torq.testing.comparison import compare_test_results
from torq.testing.onnx import generate_onnx_layer_from_file
from torq.testing.iree import llvmcpu_reference_results
from torq.testing.iree import  list_files

from torq.testing.versioned_fixtures import versioned_cached_data_fixture

'''
Any onnx model under tests/testdata/dev_ops, tests/testdata/onnx_models
could be tested with full model and all their layers

To see all the test cases:
pytest tests/test_onnx_model.py -v -s --collect-only

run the full model:
pytest tests/test_onnx_model.py -v -s [filename.stem]_full_model

run layer by layer:
pytest tests/test_onnx_model.py -v -s [filename.stem]_layer_[layername]
for example: pytest tests/test_onnx_model.py -v -s mbv2.quant_layer_DequantizeLinear_306
'''

@pytest.fixture
def case_config(request, chip_config):

    next_chip = (chip_config.data['target'] != "SL2610")
    if next_chip:
        pytest.xfail("AssertionError: Nans differ")
    
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
        print(type(f))
        cases += generate_onnx_layer_from_file(f, node_groups)

    metafunc.parametrize("onnx_layer_model", cases, indirect=True, ids=[c.name for c in cases])


# Not ready for that (Nan differs)
# @pytest.mark.fpga_ci
@pytest.mark.ci
def test_onnx_model_llvmcpu_torq(request, llvmcpu_reference_results, torq_results, case_config, onnx_layer_model):
    compare_test_results(request, torq_results, llvmcpu_reference_results, case_config)
