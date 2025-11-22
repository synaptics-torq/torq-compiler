import pytest

from torq.testing.comparison import compare_test_results
from torq.testing.cases import get_test_cases_from_files
from torq.testing.iree import list_mlir_files


@pytest.fixture(params=get_test_cases_from_files(list_mlir_files("linalg_ops")))
def case_config(request):
    
    if request.param.data.name in ["rsqrt-bf16.mlir"]:
        pytest.xfail("output mismatch in bfloat16")

    return {
        "mlir_model_file": "static_mlir_model_file",
        "static_mlir_model_file": request.param.data,
        "input_data": "tweaked_random_input_data",
        "comparison_config": "comparison_config_from_mlir",
        "torq_compiler_options": ["--iree-input-type=linalg-torq", "--torq-css-qemu"]
    }

@pytest.mark.ci
def test_mlir_files(request, torq_results, llvmcpu_reference_results, case_config):
    compare_test_results(request, torq_results, llvmcpu_reference_results, case_config)
