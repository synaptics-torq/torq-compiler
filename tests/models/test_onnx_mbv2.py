import pytest

from torq.testing.comparison import compare_test_results
from torq.testing.onnx import generate_onnx_layers_from_hf
from torq.testing.iree import llvmcpu_reference_results

from torq.testing.versioned_fixtures import versioned_hashable_object_fixture

@versioned_hashable_object_fixture
def comparison_config_relaxed():
    return {"fp_avg_tol": 0.02, "fp_max_tol": 0.02}

@versioned_hashable_object_fixture
def comparison_config_for_reduce_mean_mbv2():
    return {"fp_avg_tol": 0.02, "fp_max_tol": 0.5}

@versioned_hashable_object_fixture
def comparison_config_for_mbv2():
    return {"fp_avg_tol": 0.03, "fp_max_tol": 0.8}

@pytest.fixture
def case_config(request, runtime_hw_type, chip_config):

    torq_compiler_options = []
    if "full_model" in request.node.name:
        # Fix for bf16 clamp going to CSS
        torq_compiler_options += ["--torq-disable-css"]

    case_config_dict = {
         "onnx_model": "onnx_layer_model",
         "mlir_model_file": "onnx_mlir_model_file",
         "input_data": "tweaked_random_input_data",
         "torq_compiler_options": torq_compiler_options
    }

    relaxed_tolerance_tc = [
        # Max relative difference: 0.01106114499270916
        # Max absolute difference: 0.000152587890625
        # Number of differences: 1 out of 1000 [0.10%]        # Accuracy error due to the way comparison is handling close to 0 values
        "layer_Gemm_64"
    ]
    if any(s in request.node.nodeid for s in relaxed_tolerance_tc):
        case_config_dict["comparison_config"] = "comparison_config_relaxed"

    if "layer_ReduceMean_62" in request.node.name:
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


@pytest.mark.fpga_ci
@pytest.mark.ci
def test_onnx_model_llvmcpu_torq(request, llvmcpu_reference_results, torq_results, case_config, onnx_layer_model):
    compare_test_results(request, torq_results, llvmcpu_reference_results, case_config)
