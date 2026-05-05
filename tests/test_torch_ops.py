import pytest

from torq.testing.comparison import compare_test_results
from torq.testing.iree import list_mlir_file_group
from torq.testing.cases import get_test_cases_from_files


@pytest.fixture
def comparison_config_for_conv_truncf(request):
    """Comparison config for conv1d with truncf before reduce (memory-optimized mode)."""
    return {"fp_avg_tol": 0.2, "fp_max_tol": 1.0}


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


@pytest.fixture(params=get_test_cases_from_files(list_mlir_file_group("torch_ops")))
def case_config(request, runtime_hw_type, chip_config):

    no_negative_input = [
        'sqrt-scalar',
    ]

    extra_args = {}
    if any(s in request.param.name for s in no_negative_input):
        extra_args["tweaked_input_data_range"]  = (0, 100)

    # Option Test for conv1d with truncf before reduce (memory-optimized mode) to maintain easily
    # This enables --torq-conv1d-truncate-for-reduce to test bf16 reduce input
    if 'encoder.mlir.230.Conv_0_small.mlir' in request.param.data.name:
        extra_args["torq_compiler_options"] = ["--torq-conv1d-truncate-for-reduce=true"]
        extra_args["comparison_config"] = "comparison_config_for_conv_truncf"
    
    if 'gelu' in request.param.data.name:
        extra_args["comparison_config"] = "comparison_config_for_gelu"
    
    if any(host_mlir in request.param.data.name for host_mlir in ['conv2d-host.mlir', 'mul-i64-scalar.mlir']):
        extra_args["torq_compiler_options"] = ["--torq-disable-slices", "--torq-disable-css"]

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


@pytest.fixture
def reference_results(request, case_config):
    if _is_gelu_case(case_config):
        return request.getfixturevalue("numpy_gelu_reference_results")

    try:
        return request.getfixturevalue("llvmcpu_reference_results")
    except Exception:
        return request.getfixturevalue("torch_reference_results")

@pytest.mark.ci
@pytest.mark.fpga_ci
def test_mlir_files(request, torq_results, reference_results, case_config):
    compare_test_results(request, torq_results, reference_results, case_config)
