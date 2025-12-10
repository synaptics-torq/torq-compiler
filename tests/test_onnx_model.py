import pytest
import onnx
from onnx import shape_inference

from torq.testing.comparison import compare_test_results
from torq.testing.onnx import generate_onnx_layers_from_model, get_full_model
from torq.testing.iree import llvmcpu_reference_results
from torq.testing.cases import Case
from torq.testing.iree import  list_files_without_extras

from torq.testing.versioned_fixtures import versioned_cached_data_fixture

'''
Any onnx model under tests/testdata/dev_ops could be tested with full model and all layers

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

    failed_str = [
    ]

    if any(s in request.node.name for s in failed_str):
        pytest.xfail("failing test or skipped for now")
    
    case_config_dict = {
         "onnx_model": "onnx_layer_model",
         "mlir_model_file": "onnx_mlir_model_file",
         "input_data": "tweaked_random_input_data",
    }

    return case_config_dict

def pytest_generate_tests(metafunc):

    files = list_files_without_extras("dev_ops")

    if not files:
        return

    cases = []
    for f in files:
        m = get_full_model(str(f))

        layers = generate_onnx_layers_from_model(m)
        cases += [Case(f"{f.stem}_{key}", layer) for key, layer in layers.items()] + [ Case(f"{f.stem}_full_model", m) ]

    metafunc.parametrize("onnx_layer_model", cases, indirect=True, ids=[c.name for c in cases])


def test_onnx_model_llvmcpu_torq(request, llvmcpu_reference_results, torq_results, case_config, onnx_layer_model):
    compare_test_results(request, torq_results, llvmcpu_reference_results, case_config)
