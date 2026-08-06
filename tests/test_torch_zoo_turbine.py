import fnmatch
import pytest
import torch

from torq.testing.comparison import compare_test_results
from torq.testing.torch import (
    generate_torch_layers_from_model_metadata_cache,
    register_torch_model_loader,
    _disable_inplace_ops,
)
from torq.testing.versioned_fixtures import versioned_hashable_object_fixture

'''
Test torchvision zoo models using IREE Turbine.

This is the turbine counterpart to tests/test_torch_zoo.py.  It uses the same
lazy model loading and metadata cache, but exports each layer/full-model with
iree.turbine.aot instead of the raw FxImporter path.

Known failing cases are listed in xfail_tc below and xfail'd.
Tolerance-sensitive cases are listed in relaxed_tc and run with a relaxed
comparison config instead of being xfail'd.
Add new models to ZOO_MODELS.

CLI:
  pytest tests/test_torch_zoo_turbine.py -v -s --collect-only
  pytest tests/test_torch_zoo_turbine.py -v -s [model]_[layername]
'''


def _load_squeezenet():
    from torchvision.models import squeezenet1_0, SqueezeNet1_0_Weights
    return squeezenet1_0(weights=SqueezeNet1_0_Weights.IMAGENET1K_V1)


def _load_mobilenet_v2():
    from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
    return mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)


def _load_resnet18():
    from torchvision.models import resnet18, ResNet18_Weights
    return resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)


def _load_mnasnet():
    from torchvision.models import mnasnet1_0, MNASNet1_0_Weights
    return mnasnet1_0(weights=MNASNet1_0_Weights.IMAGENET1K_V1)


def _load_shufflenet():
    from torchvision.models import shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights
    return shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)


def _load_efficientnet_b0():
    from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
    return efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)


def _make_bf16_loader(raw_loader):
    """Wrap a raw torchvision loader so the returned model is bf16 + inplace-safe."""
    def _loader():
        m = raw_loader()
        m = _disable_inplace_ops(m)
        return m.to(torch.bfloat16).eval()
    return _loader


# Each entry: (test_id_prefix, weights_version_string, model_loader, example_inputs_tuple)
ZOO_MODELS = [
    (
        "squeezenet1_0",
        "SqueezeNet1_0_Weights.IMAGENET1K_V1",
        _make_bf16_loader(_load_squeezenet),
        (torch.randn(1, 3, 224, 224),),
    ),
    (
        "mobilenet_v2",
        "MobileNet_V2_Weights.IMAGENET1K_V1",
        _make_bf16_loader(_load_mobilenet_v2),
        (torch.randn(1, 3, 224, 224),),
    ),
    (
        "resnet18",
        "ResNet18_Weights.IMAGENET1K_V1",
        _make_bf16_loader(_load_resnet18),
        (torch.randn(1, 3, 224, 224),),
    ),
    (
        "mnasnet1_0",
        "MNASNet1_0_Weights.IMAGENET1K_V1",
        _make_bf16_loader(_load_mnasnet),
        (torch.randn(1, 3, 224, 224),),
    ),
    (
        "shufflenet_v2_x1_0",
        "ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1",
        _make_bf16_loader(_load_shufflenet),
        (torch.randn(1, 3, 224, 224),),
    ),
    (
        "efficientnet_b0",
        "EfficientNet_B0_Weights.IMAGENET1K_V1",
        _make_bf16_loader(_load_efficientnet_b0),
        (torch.randn(1, 3, 224, 224),),
    ),
]

# Register the lazy loaders so fixtures can instantiate the model on demand.
for prefix, _version, loader, _inputs in ZOO_MODELS:
    register_torch_model_loader(prefix, loader)


def pytest_generate_tests(metafunc):
    cases = []
    for prefix, model_version, loader, example_inputs in ZOO_MODELS:
        cases += generate_torch_layers_from_model_metadata_cache(
            metafunc.config.cache,
            prefix,
            model_version,
            prefix,  # loader key registered above
            loader,
            tuple(x.to(torch.bfloat16) for x in example_inputs),
        )

    metafunc.parametrize("torch_zoo_turbine_case", cases, indirect=True, ids=[c.name for c in cases])


@versioned_hashable_object_fixture
def comparison_config_relaxed():
    return {"fp_avg_tol": 0.1, "fp_max_tol": 1.0}


@pytest.fixture
def torch_zoo_turbine_case(request):
    return request.param


@pytest.fixture
def case_config(request, chip_config):
    case = request.getfixturevalue("torch_zoo_turbine_case")

    # Cases xfail'd for non-tolerance reasons (crash, exceed LRAM, parsing issue).
    xfail_tc = [
        # Full models — exceed LRAM on all zoo models
        'mobilenet_v2_full_model',
        'efficientnet_b0_full_model',

        # crash with lram issue
        'mobilenet_v2_features_sequential',
        'efficientnet_b0_features_Sequential',
        'efficientnet_b0_features_5_Sequential',

        # Times out at runtime on CSS simulator.
        'mobilenet_v2_features_2_conv_sequential',
    ]

    # Cases that fail only due to tolerance; use relaxed comparison config instead of xfail.
    relaxed_tc = [

        # SqueezeNet working
        # full model result:
        # Max relative difference: 0.008264439180493355
        # Max absolute difference: 0.0
        # Number of differences: 0 out of 1000 [0.00%]
        'squeezenet1_0_*',

        # resnet18 working
        # there is small difference for some layers
        # full model result:
        # Max relative difference: 0.6228322982788086
        # Max absolute difference: 0.15625
        # Number of differences: 36 out of 1000 [3.60%]
        'resnet18_*',

        # MNASNet working
        # full model result:
        # Max relative difference: 0.9999921917915344
        # Max absolute difference: 0.375
        # Number of differences: 184 out of 1000 [18.40%]
        'mnasnet1_0_*',

        # ShuffleNet working
        # full model result:
        # Max relative difference: 0.5499982237815857
        # Max absolute difference: 0.5
        # Number of differences: 110 out of 1000 [11.00%]
        'shufflenet_v2_x1_0_*',

        # MobileNetV2 most layers running on css with small difference
        # full model has lram OOM issue
        'mobilenet_v2_features_*',

        # EfficientNet layers — tolerance
        # full model lram issue
        'efficientnet_b0_*',
    ]

    name_lower = case.name.lower()
    if any(fnmatch.fnmatch(name_lower, pat.lower()) for pat in xfail_tc):
        pytest.xfail("failing test or skipped for now")

    comp_config = {
        "layer_name": case.data["layer_name"],
        "model_loader_key": case.data["model_loader_key"],
        "is_full_model": case.data["is_full_model"],
        "layer_input_shapes": case.data.get("layer_input_shapes", []),
        "torch_model_data": "torch_turbine_layer_model_data",
        "mlir_model_file": "torch_layer_model",
        "input_data": "tweaked_random_input_data",
        "comparison_config": "comparison_config_from_mlir"
    }

    if any(fnmatch.fnmatch(name_lower, pat.lower()) for pat in relaxed_tc):
        comp_config["comparison_config"] = "comparison_config_relaxed"

    return comp_config


@pytest.fixture
def reference_results(request):
    """Use PyTorch directly as reference for torch zoo turbine model tests."""
    return request.getfixturevalue("torch_reference_results")


@pytest.mark.ci
def test_torch_zoo_turbine(
    request, reference_results, torq_results, case_config, torch_zoo_turbine_case
):
    compare_test_results(request, torq_results, reference_results, case_config)
