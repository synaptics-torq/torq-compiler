import pytest

try:
    import jax
except ImportError:
    pytest.skip("jax is not installed", allow_module_level=True)


from torq.testing.versioned_fixtures import versioned_unhashable_object_fixture
from torq.testing.comparison import compare_test_results


@versioned_unhashable_object_fixture
def smoke_test(request):
    
    @jax.jit
    def f(x):
        return x + 1

    return f

@versioned_unhashable_object_fixture
def smoke_test_args(request):
    return (jax.core.ShapedArray((1,), jax.numpy.float32),)



@pytest.fixture
def case_config(request):
    return {"jax_function": "smoke_test", 
            "input_data": "tweaked_random_input_data",
            "jax_abstract_args": "smoke_test_args",
            "mlir_model_file": "jax_mlir_model_file"}


@pytest.mark.ci
def test_jax(request, llvmcpu_reference_results, torq_results, case_config):
    compare_test_results(request, torq_results, llvmcpu_reference_results, case_config)    
