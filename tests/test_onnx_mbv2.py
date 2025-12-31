import pytest

from torq.testing.comparison import compare_test_results
from torq.testing.onnx import generate_onnx_layers_from_hf
from torq.testing.iree import llvmcpu_reference_results

from torq.testing.versioned_fixtures import versioned_cached_data_fixture

@versioned_cached_data_fixture
def comparison_config_for_reduce_mean_mbv2(request):
    return {"fp_avg_tol": 0.03, "fp_max_tol": 0.31}

@versioned_cached_data_fixture
def comparison_config_for_mbv2(request):
    return {"fp_avg_tol": 0.03, "fp_max_tol": 0.8}

@pytest.fixture
def case_config(request, runtime_hw_type, chip_config):

    failed_tc = []

    aws_fpga = (runtime_hw_type.data == "aws_fpga")
    if aws_fpga:
        failed_tc += [
            # Compile time timeout
            "layer_Add_9",
            # because of the above
            "full_model"
        ]

    next_chip = (chip_config.data['target'] != "SL2610")
    if next_chip:
        failed_tc += [
            # Compile time timeout
            "layer_Conv_3",
            # Accuracy issue on a specific
            "layer_Conv_0",
            # because of the above
            "full_model"
        ]

    if any(s in request.node.name for s in failed_tc):
        pytest.xfail("failing test or skipped for now")

    case_config_dict = {
         "onnx_model": "onnx_layer_model",
         "mlir_model_file": "onnx_mlir_model_file",
         "input_data": "tweaked_random_input_data",
    }

    if "layer_ReduceMean" in request.node.name:
        case_config_dict["comparison_config"] = "comparison_config_for_reduce_mean_mbv2"

    if "full_model" in request.node.name:
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
@pytest.mark.fpga_ci
@pytest.mark.ci
def test_onnx_model_llvmcpu_torq(request, llvmcpu_reference_results, torq_results, case_config, onnx_layer_model):
    compare_test_results(request, torq_results, llvmcpu_reference_results, case_config)
