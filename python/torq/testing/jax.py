import pytest
from torq.testing.versioned_fixtures import versioned_generated_file_fixture

try:
    import jax
except ImportError:
    pytest.skip("jax is not installed", allow_module_level=True)


@pytest.fixture
def jax_abstract_args(request, case_config):
    return request.getfixturevalue(case_config["jax_abstract_args"])


@pytest.fixture
def jax_function(request, case_config):
    return request.getfixturevalue(case_config["jax_function"])


@versioned_generated_file_fixture("mlir")
def jax_mlir_model_file(request, versioned_file, jax_function, jax_abstract_args):
    
    lowered = jax.jit(jax_function).lower(*jax_abstract_args)

    # TODO: export as mlirb instead of text
    hlo_text = lowered.as_text()

    with open(versioned_file, "w") as f:
        f.write(hlo_text)
