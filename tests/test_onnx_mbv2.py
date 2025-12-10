import pytest
import onnx
from onnx import shape_inference

from torq.testing.comparison import compare_test_results
from torq.testing.onnx import generate_onnx_layers_from_model, get_full_model
from torq.testing.iree import llvmcpu_reference_results
from torq.testing.hf import get_hf_model_file
from torq.testing.cases import Case

from torq.testing.versioned_fixtures import versioned_cached_data_fixture

@versioned_cached_data_fixture
def comparison_config_for_mbv2(request):
    return {"fp_avg_tol": 0.03, "fp_max_tol": 0.31}


@pytest.fixture
def case_config(request, chip_config):

    next_chip = (chip_config.data['target'] != "SL2610")
    if next_chip:
        pytest.xfail("AssertionError: Nans differ")

    failed_str = [
        # crash as dw not supported, fallback to conv2d with tiling issue
        # basically wrong in lowering logic
        "layer_Conv_4-",
        "layer_Conv_11-",

        # wrong result as torq fc kernel hasn't support bf16
        "layer_Gemm", 

        # wrong result
        "layer_Conv_3-",
        "layer_Conv_6-",
        "layer_Conv_10-",
        "layer_Conv_13",
        "layer_Conv_17",
        "layer_Conv_21",
        "layer_Conv_24",
        "layer_Conv_28",
        "layer_Conv_32",
        "layer_Conv_36",
        "layer_Conv_39",
        "layer_Conv_43",
        "layer_Conv_47",
        "layer_Conv_50",
        "layer_Conv_54",
        "layer_Conv_58",
        "layer_Conv_61",

        # full model fail because of above op issue
        # full model has total 62 layers as we group some ops together (conv2d + clip)
        "full_model"
    ]

    if any(s in request.node.name for s in failed_str):
        pytest.xfail("failing test or skipped for now")
    
    case_config_dict = {
         "onnx_model": "onnx_layer_model",
         "mlir_model_file": "onnx_mlir_model_file",
         "input_data": "tweaked_random_input_data",
    }

    if "layer_ReduceMean" in request.node.name:
        case_config_dict["comparison_config"] = "comparison_config_for_mbv2"

    return case_config_dict


def pytest_generate_tests(metafunc):

    model_file = get_hf_model_file(metafunc.config.cache,  "Synaptics/torch_models", "mbv2-bf16.onnx")
    model = get_full_model(model_file)

    # Define groups of nodes (by op_type) that should be split together
    node_groups = [
        ['Conv', 'Clip'],
    ]
    layers = generate_onnx_layers_from_model(model, node_groups)

    cases = [Case(key, layer) for key, layer in layers.items()] + [ Case("full_model", model) ]
    metafunc.parametrize("onnx_layer_model", cases, indirect=True)


# Not ready for that (Nan differs)
#  @pytest.mark.fpga_ci
@pytest.mark.ci
def test_onnx_model_llvmcpu_torq(request, llvmcpu_reference_results, torq_results, case_config, onnx_layer_model):
    compare_test_results(request, torq_results, llvmcpu_reference_results, case_config)
