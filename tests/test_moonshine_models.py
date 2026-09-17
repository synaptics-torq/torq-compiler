import numpy as np
import pytest

from torq.testing.comparison import compare_test_results
from torq.testing.onnx import generate_onnx_layers_from_model, get_full_model, numpy_reference_results, has_bf16_matmul
from torq.testing.iree import llvmcpu_reference_results
from torq.lab.pipeline.io import get_dtype
from torq.testing.hf import get_hf_model_file
from torq.testing.cases import Case

from torq.testing.versioned_fixtures import versioned_hashable_object_fixture, versioned_cached_data_fixture


@versioned_hashable_object_fixture
def comparison_config_relaxed():
    return {"fp_avg_tol": 0.02, "fp_max_tol": 1.0}

@versioned_hashable_object_fixture
def comparison_config_full_model():
    # The comparison's relative-difference metric is |e-o|/(|e|+|o|+eps), which is
    # mathematically bounded in [0, 1).  A fp_max_tol of 1.0 (as in
    # comparison_config_relaxed) therefore can never be exceeded, which silently
    # disables the check.  Keep fp_max_tol strictly below 1.0 so the full-model
    # comparison actually gates on output correctness.  Pair the relative gate with
    # an absolute gate (fp_abs_tol_frac, a few bf16 ULP of the tensor's range) so the
    # near-zero cancellation outputs that bf16 matmul produces are not flagged, while
    # broad numerical drift still fails.
    return {"fp_avg_tol": 0.02, "fp_max_tol": 0.2, "use_abs_tol_gate": True, "fp_abs_tol_frac": 0.02, "allowed_wrong": 0.001}

@versioned_cached_data_fixture
def moonshine_decoder_input_data(request, tweaked_random_input_data, mlir_io_spec):
    """Input data for the moonshine decoder full model.

    tweaked_random_input_data randomizes every input, but the decoder's
    `current_len` input is a control index (the current decode position), not free
    data.  An out-of-range value (the default tweaked range is (-40, 40)) makes the
    causal attention mask, the rotary position, and the KV-cache scatter degenerate:
    the `present.*` outputs collapse to a pass-through of the past KV inputs and the
    logits become meaningless and diverge between ONNXRuntime and the NPU.  Replace
    the (single) integer input with a valid sequence position so the comparison
    exercises the real decode path.
    """
    valid_current_len = 5  # any position in [0, decoder KV seq length)
    data = [d.copy() for d in tweaked_random_input_data]
    for i, tensor_type in enumerate(mlir_io_spec.inputs):
        if np.issubdtype(get_dtype(tensor_type.fmt), np.integer):
            data[i] = np.full(tensor_type.shape, valid_current_len, dtype=data[i].dtype)
    return data

@versioned_hashable_object_fixture
def comparison_config_for_tanh():
    return {"fp_avg_tol": 0.033, "fp_max_tol": 0.063}

@versioned_hashable_object_fixture
def comparison_config_for_activation_noise():
    return {"fp_avg_tol": 0.05, "fp_max_tol": 0.2, "allowed_wrong": 0.001}

@versioned_hashable_object_fixture
def comparison_config_for_bf16_gemm():
    # The TORQ MatMul kernel runs at bf16 precision: it emits bf16 outputs and carries
    # ~1 ULP of accumulation noise relative to the fp32 ONNX reference.  On near-zero
    # cancellation outputs that small absolute error becomes a large *relative* error, so
    # gate on relative AND absolute difference (fp_abs_tol_frac = a few bf16 ULP of the
    # tensor's range) instead of blanket-allowing a fraction of grossly-wrong elements.
    return {"fp_avg_tol": 0.05, "fp_max_tol": 0.2, "use_abs_tol_gate": True, "fp_abs_tol_frac": 0.02, "allowed_wrong": 0.0}

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
    torq_compiler_options = ["--torq-convert-dtypes", "--torq-convert-io-dtype"]
    if any(s in request.node.name for s in nss_layers):
        torq_compiler_options += ["--torq-disable-css", "--torq-disable-host"]

    comp_config["torq_compiler_options"] = torq_compiler_options

    if "encoder_float_layer_Tanh_13-" in request.node.name:
        comp_config["comparison_config"] = "comparison_config_for_tanh"

    activation_noise_cases = [
        "decoder_float_layer_Sigmoid_107",
        "encoder_float_layer_InstanceNormalization_15",
        "encoder_float_layer_Softmax_112",
    ]
    if any(case in request.node.name for case in activation_noise_cases):
        comp_config["comparison_config"] = "comparison_config_for_activation_noise"

    # MatMul layers that go through TileAndFuse's K-split accumulate the K-chunk
    # partial sums in bf16. Near-zero outputs (cancellation) then show a large
    # relative error but a tiny absolute error, so use the bf16 abs-gate config.
    bf16_gemm_cases = [
        "decoder_float_layer_Gemm_103",
        "decoder_float_layer_Gemm_111",
        "encoder_float_layer_Gemm_127",
        "encoder_float_layer_Gemm_135",
        "encoder_float_layer_MatMul_36",
        "encoder_float_layer_MatMul_53",
    ]
    if any(case in request.node.name for case in bf16_gemm_cases):
        comp_config["comparison_config"] = "comparison_config_for_bf16_gemm"

    if "full_model" in request.node.name:
        comp_config["comparison_config"] = "comparison_config_full_model"

    # The decoder's `current_len` control input must be a valid position, so the
    # decoder full model uses a dedicated input fixture instead of fully random data.
    # Also narrow the float range from the default (-40, 40) to a more realistic
    # activation magnitude so bf16 precision loss isn't artificially amplified.
    if "decoder" in request.node.name and "full_model" in request.node.name:
        comp_config["input_data"] = "moonshine_decoder_input_data"
        comp_config["tweaked_input_data_range"] = (-2, 2)

    return comp_config


def pytest_generate_tests(metafunc):
    testtypes = [
        "float",
    #   "quantized"
    ]
    testcases = ["decoder", "encoder"]

    quantize = metafunc.config.getoption("--quantize", default=False)

    cases = []
    for ttype in testtypes:
        for t in testcases:
            testbf16 = f"onnx/{ttype}/{t}.onnx"
            model_file = get_hf_model_file(metafunc.config.cache,  "Synaptics/Moonshine", testbf16)
            model = get_full_model(model_file)
            layers = generate_onnx_layers_from_model(model, quantize=quantize)

            cases += [Case(f"{t}_{ttype}_{key}", layer) for key, layer in layers.items()] + [ Case(f"{t}_{ttype}_full_model", model) ]

    skip_cmodel = [
        # timeout on CModel
        "encoder_float_full_model",
    ]
    params = []
    for case in cases:
        marks = (
            (pytest.mark.fpga_ci, )
            if case.name in skip_cmodel
            else (pytest.mark.ci, pytest.mark.fpga_ci, )
        )
        params.append(pytest.param(case, marks=marks))

    metafunc.parametrize("onnx_layer_model", params, indirect=True)


@versioned_hashable_object_fixture
def onnx_fake_quantize_config():
    # Always fake-quantize FP32/INT64 weights to BF16/INT32 for these models.
    return {"fake_quantize": True}


@pytest.fixture
def reference_results(request, onnx_layer_model):
    return request.getfixturevalue("onnx_reference_results")


def test_onnx_model_torq(request, reference_results, torq_results, case_config, onnx_layer_model):
    compare_test_results(request, torq_results, reference_results, case_config)
