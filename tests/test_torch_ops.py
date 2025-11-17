import pytest

from torq.testing.comparison import compare_test_results
from torq.testing.iree import list_mlir_file_group
from torq.testing.cases import get_test_cases_from_files


@pytest.fixture(params=get_test_cases_from_files(list_mlir_file_group("torch_ops")))
def case_config(request):

    if request.param.data.name in ["equal.mlir", "instancenorm.mlir", 
                                    "0135_ReduceMean__layers.0_post_attention_layernorm_ReduceMean.mlir"]:
        pytest.xfail("not implemented yet")

    return {
        "mlir_model_file": "static_mlir_model_file",
        "static_mlir_model_file": request.param.data,
        "input_data": "tweaked_random_input_data",
        "torq_compiler_options": ["--torq-css-qemu"]
    }

@pytest.mark.ci
def test_mlir_files(request, torq_results, llvmcpu_reference_results, case_config):    
    compare_test_results(request, torq_results, llvmcpu_reference_results, case_config)
