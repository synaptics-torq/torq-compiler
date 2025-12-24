import pytest

from torq.testing.comparison import compare_test_results
from torq.testing.onnx import generate_onnx_layers_from_hf
from torq.testing.iree import llvmcpu_reference_results

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
        # wrong result as torq fc kernel hasn't support bf16
        "layer_Gemm", 

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

    # Define groups of nodes (by op_type) that should be split together
    node_groups = [
        ['Conv', 'Clip'],
    ]

    cases = generate_onnx_layers_from_hf(metafunc.config.cache,
            "Synaptics/torch_models", "mbv2-bf16.onnx", node_groups)

    metafunc.parametrize("onnx_layer_model", cases, indirect=True)


# Not ready for that (Nan differs)
#  @pytest.mark.fpga_ci
@pytest.mark.ci
def test_onnx_model_llvmcpu_torq(request, llvmcpu_reference_results, torq_results, case_config, onnx_layer_model):
    compare_test_results(request, torq_results, llvmcpu_reference_results, case_config)