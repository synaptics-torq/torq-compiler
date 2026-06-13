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

    Block quantization (Q8_0) introduces rounding at block boundaries that
    amplifies relative error for small output values. 12-13% of elements
    typically exceed the default fp_max_tol.
    """
    return {"fp_max_tol": 0.5, "allowed_wrong": 0.15}


@pytest.fixture
def comparison_config_for_instancenorm(request):
    """Comparison config for InstanceNormalization.

    The normalized bf16 output carries ~1 bf16 ULP of error; against the fp32
    numpy reference a few near-zero elements just exceed the default fp_max_tol.
    """
    return {"fp_max_tol": 2e-2, "allowed_wrong": 1e-3}


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


@pytest.fixture
def reference_results(request, case_config):
    if _is_gelu_case(case_config):
        return request.getfixturevalue("numpy_gelu_reference_results")

    if _is_global_average_pool_case(case_config):
        return request.getfixturevalue("numpy_global_average_pool_reference_results")

    if _is_instancenorm_case(case_config):
        return request.getfixturevalue("numpy_instancenorm_reference_results")

    try:
        return request.getfixturevalue("llvmcpu_reference_results")
    except Exception:
        return request.getfixturevalue("torch_reference_results")

@pytest.mark.ci
@pytest.mark.fpga_ci
def test_mlir_files(request, torq_results, reference_results, case_config):
    compare_test_results(request, torq_results, reference_results, case_config)
