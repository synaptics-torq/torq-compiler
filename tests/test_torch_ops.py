import pytest

from torq.testing.comparison import compare_test_results
from torq.testing.iree import list_mlir_file_group
from torq.testing.cases import get_test_cases_from_files


@pytest.fixture
def comparison_config_for_conv_truncf(request):
    """Comparison config for conv1d with truncf before reduce (memory-optimized mode)."""
    return {"fp_avg_tol": 0.2, "fp_max_tol": 1.0}

@pytest.fixture
def comparison_config_for_localavgpool(request):
    return {"fp_avg_tol": 0.2, "fp_max_tol": 1.004}

@pytest.fixture
def comparison_config_for_gelu(request):
    """Comparison config for GELU.

    GELU can produce tiny values in the far negative tail. The observed
    mismatch was:

        expected = -1.611188e-7, observed = -0.0, abs_diff = 1.611188e-7

    This is safe for GELU correctness because the mismatch is in the tail where
    the true GELU result is already effectively zero.
    """
    return {"epsilon": 2e-5}


@pytest.fixture
def comparison_config_for_matmul_dql(request):
    """Comparison config for block-quantized matmul (DequantizeLinear + MatMul).

    Block quantization introduces rounding at block boundaries that
    amplifies relative error for small output values.
    """
    return {"fp_max_tol": 0.5, "allowed_wrong": 0.2}


@pytest.fixture
def comparison_config_for_instancenorm(request):
    """Comparison config for InstanceNormalization.

    The normalized bf16 output carries ~1 bf16 ULP of error; against the fp32
    numpy reference a few near-zero elements just exceed the default fp_max_tol.
    """
    return {"fp_max_tol": 2e-2, "allowed_wrong": 1e-3}


@pytest.fixture
def comparison_config_for_qdq_dequant(request):
    """int8 QDQ dequant zp regression (deq_min_i8, qdq_min_i8): residual is bf16 MulOp rounding."""
    return {"fp_max_tol": 0.2, "allowed_wrong": 0.06}


@pytest.fixture
def comparison_config_for_qdq_conv_tiled(request):
    """Tiled int8 QDQ conv W-shift regression (conv1d_307_qdq_tiled): residual is bf16 requant rounding."""
    return {"fp_max_tol": 0.2, "allowed_wrong": 0.02}


@pytest.fixture
def comparison_config_for_qdq_conv_295(request):
    """Legacy-path tiled int8 QDQ conv (conv1d_295_qdq_tiled, kw=11). Its 1x64x6401 output spans a
    wide dynamic range (~1.0 down to ~1e-4), so the near-zero tail's bf16 requant rounding trips the
    relative-error metric on a few percent of elements, and the exact fraction varies by chip requant
    (e.g. ~2.0% on sr200 vs less on sl2610). Loosened accordingly -- a real W-shift regression is ~80%
    wrong, nowhere near this floor."""
    return {"fp_max_tol": 0.2, "allowed_wrong": 0.05}


@pytest.fixture
def comparison_config_for_qdq_izp(request):
    """int8 QDQ conv input-zero-point bias-fold regression (conv_izp_i8): residual is bf16 requant rounding."""
    return {"fp_max_tol": 0.2, "allowed_wrong": 0.06}


@pytest.fixture(params=get_test_cases_from_files(list_mlir_file_group("torch_ops")))
def case_config(request, runtime_hw_type, chip_config):

    no_negative_input = [
        'sqrt-',
    ]

    extra_args = {}
    if any(s in request.param.name for s in no_negative_input):
        extra_args["tweaked_input_data_range"]  = (0, 100)

    # sin/cos are optimized for RoPE, which only runs on (often small)
    # positive input.  For that reason, negative inputs are not quite
    # as accurate as they could be.
    if 'sin-exact' in request.param.name:
        extra_args["tweaked_input_data_range"] = 0, 3
    if 'cos-exact' in request.param.name:
        extra_args["tweaked_input_data_range"] = 0, 1.5
    if 'sin-coarse' in request.param.name:
        extra_args["tweaked_input_data_range"] = 0, 12
    if 'cos-coarse' in request.param.name:
        extra_args["tweaked_input_data_range"] = 0, 12

    # The K-split matmul case accumulates chunk partials in bf16; keep inputs
    # positive so cancellation noise doesn't dominate the comparison (the
    # TORQ_* tolerance directives in the .mlir cover the residual rounding).
    if 'matmul-1x128x3072x768' in request.param.data.name:
        extra_args["tweaked_input_data_range"] = 0, 2

    no_slicing_tc = [
        #  error: failed to allocate LRAM addresses
        "matmul_dql_q8_0",
        "matmul_dql_q4_0",
        # assertion for next 2 in function LData, file Kernel.cpp, line 824
        #  (elementType != DType::none && "Invalid elemen= DType::none)
        "topk-1x2100-bf16",
        "topk-1x2100-indices-only-bf16"
    ]
    extra_args["torq_compiler_options"] = []
    if any(s in request.param.data.name for s in no_slicing_tc):
        extra_args["torq_compiler_options"].append("--torq-disable-slicing")


    # Option Test for conv1d with truncf before reduce (memory-optimized mode) to maintain easily
    # This enables --torq-conv1d-truncate-for-reduce to test bf16 reduce input
    if 'encoder.mlir.230.Conv_0_small.mlir' in request.param.data.name:
        extra_args["torq_compiler_options"] = ["--torq-conv1d-truncate-for-reduce=true"]
        extra_args["comparison_config"] = "comparison_config_for_conv_truncf"

    if 'localavgpool.mlir' in request.param.data.name:
        extra_args["comparison_config"] = "comparison_config_for_localavgpool"
    
    if 'gelu' in request.param.data.name:
        extra_args["comparison_config"] = "comparison_config_for_gelu"

    if 'matmul_dql' in request.param.data.name:
        extra_args["comparison_config"] = "comparison_config_for_matmul_dql"

    if 'instancenorm' in request.param.data.name:
        extra_args["comparison_config"] = "comparison_config_for_instancenorm"

    # Force the Conv1D-as-matmul -> fully_connected lowering tests to run on
    # the NSS/slice path so they fail loudly if a future change silently routes
    # them to the host/CSS fallback.
    if 'conv1d_matmul_bf16_' in request.param.data.name:
        extra_args["torq_compiler_options"] = ["--torq-disable-host", "--torq-disable-css"]
    
    if any(host_mlir in request.param.data.name for host_mlir in ['conv2d-host.mlir', 'mul-i64-scalar.mlir', 'constantshape.mlir']):
        extra_args["torq_compiler_options"] = ["--torq-disable-slices", "--torq-disable-css"]

    if 'softmax-1x2xbf16.mlir' in request.param.data.name:
        extra_args["torq_compiler_options"] = ["--torq-disable-css", "--torq-disable-host"]

    # Force the bf16 elementwise add onto the NSS/slice path so it fails loudly if a
    # future change stops lowering bf16 add via AddOpPattern (createBf16Add -> torq_hl.add).
    # (f32 two-tensor add is intentionally not lowered to NSS: the bf16-width data path
    # mis-strides f32 inputs and produces garbage.)
    if 'add-nss-25x511-bf16.mlir' in request.param.data.name:
        extra_args["torq_compiler_options"] = ["--torq-disable-host", "--torq-disable-css"]

    # Force the dynamic-indices gather onto the NSS/slice path so it fails loudly if
    # the value copy stops lowering to torq_hl.gather. CSS is left enabled on purpose:
    # the ONNX Gather index math (negative-index normalization on i64 plus the
    # i64->i32 demote) has no NSS data path and must run on CSS. Adding
    # --torq-disable-css would orphan those scalar ops and break serialization.
    if 'gather-dynamic-indices' in request.param.data.name:
        extra_args["torq_compiler_options"] = ["--torq-disable-host"]

    # int8 QDQ dequant zp regression: force onto NSS so a dropped-zp regression fails loudly.
    if any(s in request.param.data.name for s in ['deq_min_i8', 'qdq_min_i8']):
        extra_args["torq_compiler_options"] = ["--torq-disable-host", "--torq-disable-css"]
        extra_args["comparison_config"] = "comparison_config_for_qdq_dequant"

    # Tiled int8 QDQ conv W-shift regression: force onto NSS so the wide conv tiles along W.
    # Two variants guard the two conv lowering paths: conv1d_307 (kw=7) goes through the EK
    # kernel (Conv2DToHw.cpp); conv1d_295 (kw=11) fails hasEkLoweringConv (kw>7) and takes the
    # legacy Conv2DPattern.cpp path, whose baseOffset W-correction is what the fix restores.
    # Without that fix conv1d_295 is ~80% wrong; keep both so neither path can regress silently.
    if 'conv1d_307_qdq_tiled' in request.param.data.name:
        extra_args["torq_compiler_options"] = ["--torq-disable-host", "--torq-disable-css"]
        extra_args["comparison_config"] = "comparison_config_for_qdq_conv_tiled"

    if 'conv1d_295_qdq_tiled' in request.param.data.name:
        extra_args["torq_compiler_options"] = ["--torq-disable-host", "--torq-disable-css"]
        extra_args["comparison_config"] = "comparison_config_for_qdq_conv_295"

    # int8 QDQ conv input-zero-point regression: force onto NSS. The nonzero izp (si8 -23)
    # must fold into the bias via computeInputZpCorrection (NSS hardware does not subtract
    # izp); if that regresses, the NSS output diverges from the llvmcpu reference and fails.
    if 'conv_izp_i8' in request.param.data.name:
        extra_args["torq_compiler_options"] = ["--torq-disable-host", "--torq-disable-css"]
        extra_args["comparison_config"] = "comparison_config_for_qdq_izp"

    return {
        "mlir_model_file": "static_mlir_model_file",
        "static_mlir_model_file": request.param.data,
        "input_data": "tweaked_random_input_data",
        "comparison_config": "comparison_config_from_mlir",
        **extra_args
    }


def _is_gelu_case(case_config):
    mlir_file = case_config.get("static_mlir_model_file")
    return mlir_file is not None and "gelu" in mlir_file.name.lower()


def _is_global_average_pool_case(case_config):
    mlir_file = case_config.get("static_mlir_model_file")
    return mlir_file is not None and "globalaveragepool" in mlir_file.name.lower()


def _is_instancenorm_case(case_config):
    mlir_file = case_config.get("static_mlir_model_file")
    return mlir_file is not None and "instancenorm" in mlir_file.name.lower()


def _is_const_weight_matmul_case(case_config):
    mlir_file = case_config.get("static_mlir_model_file")
    return mlir_file is not None and "matmul-1x128x" in mlir_file.name.lower()


@pytest.fixture
def reference_results(request, case_config):
    if _is_gelu_case(case_config):
        return request.getfixturevalue("numpy_gelu_reference_results")

    if _is_global_average_pool_case(case_config):
        return request.getfixturevalue("numpy_global_average_pool_reference_results")

    if _is_instancenorm_case(case_config):
        return request.getfixturevalue("numpy_instancenorm_reference_results")

    if _is_const_weight_matmul_case(case_config):
        return request.getfixturevalue("numpy_matmul_const_weight_reference_results")

    try:
        return request.getfixturevalue("llvmcpu_reference_results")
    except Exception:
        return request.getfixturevalue("torch_reference_results")

@pytest.mark.ci
@pytest.mark.fpga_ci
def test_mlir_files(request, torq_results, reference_results, case_config):
    compare_test_results(request, torq_results, reference_results, case_config)
