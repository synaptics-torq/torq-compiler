import pytest

from torq.testing.comparison import compare_test_results
from torq.testing.torch import generate_torch_layers_from_file
from torq.testing.iree import list_files

'''
Test torch models under tests/testdata/torch_models using IREE Turbine.

This is a counterpart to tests/test_torch_model.py.  It uses the same layer
extraction logic but exports each layer/full-model with iree.turbine.aot
instead of the raw FxImporter path.

CLI examples:
  List test cases:
    pytest tests/test_torch_turbine.py -v -s --collect-only

  Run full model:
    pytest tests/test_torch_turbine.py -v -s simple_linear_full_model

  Run a layer:
    pytest tests/test_torch_turbine.py -v -s simple_linear_fc_Linear
'''


def pytest_generate_tests(metafunc):
    files = list_files("torch_models", ".pt", False)

    if not files:
        return

    cases = []
    for f in files:
        # Start with simple_linear while validating the Turbine path.
        if f.stem != "simple_linear":
            continue
        cases += generate_torch_layers_from_file(f, cache=metafunc.config.cache)

    metafunc.parametrize(
        "torch_turbine_model_case",
        cases,
        indirect=True,
        ids=[c.name for c in cases],
    )


@pytest.fixture
def torch_turbine_model_case(request):
    if not hasattr(request, "param"):
        pytest.skip("No torch turbine test cases found (.pt files under tests/testdata/torch_models)")
    return request.param


@pytest.fixture
def case_config(request, chip_config):
    case = request.getfixturevalue("torch_turbine_model_case")

    return {
        "layer_name": case.data["layer_name"],
        "model_path": case.data.get("model_path"),
        "is_full_model": case.data["is_full_model"],
        "layer_input_shapes": case.data.get("layer_input_shapes", []),
        "torch_model_data": "torch_turbine_layer_model_data",
        "mlir_model_file": "torch_layer_model",
        "input_data": "tweaked_random_input_data",
        "comparison_config": "comparison_config_from_mlir",
    }


@pytest.fixture
def reference_results(request):
    """Use PyTorch directly as reference for torch model tests."""
    return request.getfixturevalue("torch_reference_results")


@pytest.mark.ci
def test_torch_turbine_model(
    request, reference_results, torq_results, case_config, torch_turbine_model_case
):
    compare_test_results(request, torq_results, reference_results, case_config)
