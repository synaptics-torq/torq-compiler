import pytest

from torq.testing.comparison import compare_test_results
from torq.testing.onnx import (
    generate_onnx_layers_from_model,
    get_full_model,
    _has_bf16_matmul,
    _has_gelu,
)
from torq.testing.hf import get_hf_model_file
from torq.testing.cases import Case


# see test_torch_ops.py::comparison_config_for_gelu()
@pytest.fixture
def comparison_config_for_gelu(request):
    return {"epsilon": 2e-5}


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

    torq_compiler_options = []
    nss_layers = ["layer_MatMul"]
    if any(s in request.node.name for s in nss_layers):
        torq_compiler_options += ["--torq-disable-css", "--torq-disable-host"]
    comp_config["torq_compiler_options"] = torq_compiler_options

    if "Gelu" in request.node.name:
        comp_config["comparison_config"] = "comparison_config_for_gelu"

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
def reference_results(request, onnx_layer_model):
    """Select reference: numpy for bf16 MatMul/GELU, llvmcpu otherwise."""
    if _has_gelu(onnx_layer_model.data):
        return request.getfixturevalue("numpy_gelu_reference_results")
    if _has_bf16_matmul(onnx_layer_model.data):
        return request.getfixturevalue("numpy_reference_results")
    return request.getfixturevalue("llvmcpu_reference_results")


@pytest.mark.ci
@pytest.mark.fpga_ci
def test_onnx_model_torq(request, reference_results, torq_results, case_config, onnx_layer_model):
    compare_test_results(request, torq_results, reference_results, case_config)
