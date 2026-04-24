import pytest

from torq.testing.comparison import compare_test_results
from torq.testing.onnx import generate_onnx_layers_from_model, get_full_model, numpy_reference_results, _has_bf16_matmul
from torq.testing.iree import llvmcpu_reference_results
from torq.testing.hf import get_hf_model_file
from torq.testing.cases import Case


@pytest.fixture
def case_config(request, chip_config):

    no_negative_input = [
        'layer_Sqrt',
    ]

    extra_args = {}
    if any(s in request.node.name for s in no_negative_input):
        extra_args["tweaked_input_data_range"]  = (0, 100)

    next_chip = (chip_config.data['target'] != "SL2610")
    if next_chip:
        pytest.xfail("AssertionError: Nans differ")

    failed_str = [
        # llvm-cpu compile error on certain x86 systems
        "attn_block_layer_Cast_11",

        # False positive accuracy failures (llvm-cpu output is inaccurate)
        # AssertionError: Number of differences: 893 out of 2048 [43.60%]
        "attn_block_layer_Gelu_83",
        # AssertionError: Number of differences: 1 out of 4 [25.00%]
        "attn_block_layer_ReduceMean_32",
    ]

    if any(s in request.node.name for s in failed_str):
        pytest.xfail("failing test or skipped for now")

    comp_config = {
         "onnx_model": "onnx_layer_model",
         "mlir_model_file": "onnx_mlir_model_file",
         "input_data": "tweaked_random_input_data",
         "llvmcpu_compiler_options": ["--iree-global-opt-enable-early-materialization=false"],
         **extra_args
    }

    torq_compiler_options = []
    nss_layers = ["layer_MatMul"]
    if any(s in request.node.name for s in nss_layers):
        torq_compiler_options += ["--torq-disable-css", "--torq-disable-host"]
    comp_config["torq_compiler_options"] = torq_compiler_options

    return comp_config


def pytest_generate_tests(metafunc):
    gemma_components = [
        "embed_scale",
        "attn_block",
        "final_norm",
        "lm_head"
    ]

    cases = []
    for comp in gemma_components:
        model_file = get_hf_model_file(metafunc.config.cache, "Synaptics/GemmaDev", f"{comp}.onnx")
        model = get_full_model(model_file)
        layers = generate_onnx_layers_from_model(model)
        cases += [Case(f"{comp}_{key}", layer["model"]) for key, layer in layers.items()]

    metafunc.parametrize("onnx_layer_model", cases, indirect=True)


@pytest.fixture
def reference_results(request, onnx_layer_model, numpy_reference_results, llvmcpu_reference_results):
    """Select reference: numpy for bf16 MatMul, llvmcpu otherwise."""
    return numpy_reference_results if _has_bf16_matmul(onnx_layer_model.data) else llvmcpu_reference_results


@pytest.mark.ci
@pytest.mark.fpga_ci
def test_onnx_model_torq(request, reference_results, torq_results, case_config, onnx_layer_model):
    compare_test_results(request, torq_results, reference_results, case_config)
