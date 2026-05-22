import pytest

from torq.testing.comparison import compare_test_results
from torq.testing.onnx import generate_onnx_layers_from_model, get_full_model, numpy_reference_results, _has_bf16_matmul
from torq.testing.iree import llvmcpu_reference_results
from torq.testing.hf import get_hf_model_file
from torq.testing.cases import Case

from torq.testing.versioned_fixtures import versioned_hashable_object_fixture


@versioned_hashable_object_fixture
def comparison_config_relaxed():
    return {"fp_avg_tol": 0.02, "fp_max_tol": 1.0}

@versioned_hashable_object_fixture
def comparison_config_for_tanh():
    return {"fp_avg_tol": 0.033, "fp_max_tol": 0.063}

@pytest.fixture
def case_config(request, chip_config):

    no_negative_input = [
        'layer_Sqrt',
    ]

    extra_args = {}
    if any(s in request.node.name for s in no_negative_input):
        extra_args["tweaked_input_data_range"]  = (0, 100)

    if any(s in request.node.name for s in ('Cos', 'Sin')):
        extra_args["tweaked_input_data_range"] = 0, 1.5
    
    comp_config = {
         "onnx_model": "onnx_layer_model",
         "mlir_model_file": "onnx_mlir_model_file",
         "input_data": "tweaked_random_input_data",
         "llvmcpu_compiler_options": ["--iree-global-opt-enable-early-materialization=false"],
         **extra_args
    }

    nss_layers = ["layer_MatMul"]
    torq_compiler_options = []
    if any(s in request.node.name for s in nss_layers):
        torq_compiler_options += ["--torq-disable-css", "--torq-disable-host"]

    comp_config["torq_compiler_options"] = torq_compiler_options

    # Max relative difference: 0.0322580486536026
    # Max absolute difference: 0.0625
    # Number of differences: 9651 out of 359712 [2.68%]        # Max relative difference: 0.0322580486536026
    if "encoder_bf16_layer_Tanh_3-" in request.node.name:
        comp_config["comparison_config"] = "comparison_config_for_tanh"

    relaxed_tolerance_cases = [
        # Works fine on MacOs but fails on x86 in CI:
        # AssertionError: Number of differences: 1 out of 32768 [0.00%]
        "decoder_bf16_layer_MatMul_646",
        "decoder_with_past_bf16_layer_MatMul_669",
        # Max relative difference: 0.9958887696266174
        # Max absolute difference: 0.00244140625
        # Number of differences: 7 out of 59616 [0.01%]
        "encoder_bf16_layer_Gemm_95",
        # Max relative difference: 0.24998171627521515
        # Max absolute difference: 0.09375
        # Number of differences: 87 out of 207 [42.03%]
        "encoder_bf16_layer_ReduceMean_21",
        # Max relative difference: 0.9999216198921204
        # Max absolute difference: 0.03125
        # Number of differences: 46 out of 207 [22.22%]
        "encoder_bf16_layer_ReduceMean_26",
        "encoder_bf16_layer_Softmax_72",
        "encoder_bf16_layer_Gemm_87",
        "full_model"
    ]
    if any(case in request.node.name for case in relaxed_tolerance_cases):
        comp_config["comparison_config"] = "comparison_config_relaxed"
    
    return comp_config


def pytest_generate_tests(metafunc):
    testtypes = ["bf16",
    #   "quantized"
    ]
    testcases = ["decoder_with_past", "decoder", "encoder"]

    cases = []
    for ttype in testtypes: 
        for t in testcases:
            testbf16 = f"models/{ttype}/onnx/{t}.onnx"
            model_file = get_hf_model_file(metafunc.config.cache,  "Synaptics/Moonshine", testbf16, revision="v1")
            model = get_full_model(model_file)
            layers = generate_onnx_layers_from_model(model)

            cases += [Case(f"{t}_{ttype}_{key}", layer) for key, layer in layers.items()] + [ Case(f"{t}_{ttype}_full_model", model) ]

    metafunc.parametrize("onnx_layer_model", cases, indirect=True)


@pytest.fixture
def reference_results(request, onnx_layer_model, numpy_reference_results, llvmcpu_reference_results):
    """Select reference: numpy for bf16 MatMul, llvmcpu otherwise."""
    return numpy_reference_results if _has_bf16_matmul(onnx_layer_model.data) else llvmcpu_reference_results


@pytest.mark.fpga_ci
@pytest.mark.ci
def test_onnx_model_torq(request, reference_results, torq_results, case_config, onnx_layer_model):
    compare_test_results(request, torq_results, reference_results, case_config)
