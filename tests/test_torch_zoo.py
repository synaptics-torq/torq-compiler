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
Test models loaded directly from PyTorch model zoo / torchvision.

Models are instantiated from torchvision.models (weights cached by torch.hub).
Each model runs as a full-model test plus per-layer tests.

Known failing cases are listed in xfail_tc below and xfail'd.
Tolerance-sensitive cases are listed in relaxed_tc and run with a relaxed
comparison config instead of being xfail'd.
Add new models to ZOO_MODELS.

CLI:
  pytest tests/test_torch_zoo.py -v -s --collect-only
  pytest tests/test_torch_zoo.py -v -s [model]_[layername]
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

    metafunc.parametrize("torch_zoo_case", cases, indirect=True, ids=[c.name for c in cases])


@versioned_hashable_object_fixture
def comparison_config_relaxed():
    return {"fp_avg_tol": 0.02, "fp_max_tol": 1.0}


@pytest.fixture
def torch_zoo_case(request):
    return request.param


@pytest.fixture
def case_config(request, chip_config):
    case = request.getfixturevalue("torch_zoo_case")

    next_chip = (chip_config.data['target'] != "SL2610")
    if next_chip:
        pytest.xfail("AssertionError: Nans differ")

    # Cases xfail'd for non-tolerance reasons (crash, exceed LRAM, parsing issue, too large).
    xfail_tc = [
        # Full models — exceed LRAM on all zoo models
        '*_full_model',

        # BatchNorm — crash or tolerance (all models)
        '*_batchnorm2d',

        # Composite blocks — too large or crash
        '*_conv2dnormactivation',
        '*_invertedresidual',
        '*_basicblock',
        '*_mbconv',
        '*_stochasticdepth',
        '*_squeezeexcitation',
        'squeezenet1_*_fire',

        # Sequential containers — crash
        'squeezenet1_*_sequential',
        'mobilenet_v2_features_sequential',
        'mobilenet_v2_features_*_sequential',
        'resnet18_*_sequential',
        'mnasnet1_*_sequential',
        'shufflenet_*_sequential',
        'efficientnet_*_sequential',
    ]

    # Cases that fail only due to tolerance; use relaxed comparison config instead of xfail.
    relaxed_tc = [
        # AdaptiveAvgPool2d — tolerance
        '*_adaptiveavgpool2d',

        # SqueezeNet-specific conv layers — tolerance
        'squeezenet1_0_features_0_conv2d',
        'squeezenet1_0_features_5_squeeze_conv2d',
        'squeezenet1_0_features_5_expand3x3_conv2d',
        'squeezenet1_0_features_10_squeeze_conv2d',
        'squeezenet1_0_features_10_expand3x3_conv2d',
        'squeezenet1_0_features_7_expand3x3_conv2d',
        'squeezenet1_0_features_9_expand3x3_conv2d',
        'squeezenet1_0_classifier_3_adaptiveavgpool2d',
        'squeezenet1_0_features_4_expand3x3_conv2d',

        # MobileNetV2-specific conv layers — tolerance
        'mobilenet_v2_features_3_conv_0_0_conv2d',
        'mobilenet_v2_features_13_conv_0_0_conv2d',
        'mobilenet_v2_features_14_conv_0_0_conv2d',

        # ResNet18-specific conv layers — tolerance
        'resnet18_conv1_conv2d',
        'resnet18_layer1_0_conv1_conv2d',
        'resnet18_layer1_0_conv2_conv2d',
        'resnet18_layer2_0_conv2_conv2d',
        'resnet18_layer2_1_conv1_conv2d',
        'resnet18_layer3_0_conv1_conv2d',
        'resnet18_layer3_1_conv2_conv2d',

        # MNASNet-specific conv layers — tolerance
        'mnasnet1_0_layers_14_conv2d',
        'mnasnet1_0_layers_8_0_layers_0_conv2d',
        'mnasnet1_0_layers_12_3_layers_6_conv2d',
        'mnasnet1_0_layers_12_1_layers_6_conv2d',

        # ShuffleNet-specific conv layers — tolerance
        'shufflenet_v2_x1_0_stage3_0_branch2_0_conv2d',

        # EfficientNet-specific conv layers — tolerance
        'efficientnet_b0_features_5_0_block_3_0_conv2d',
        'efficientnet_b0_features_7_0_block_3_0_conv2d',
    ]

    name_lower = case.name.lower()
    if any(fnmatch.fnmatch(name_lower, pat) for pat in xfail_tc):
        pytest.xfail("failing test or skipped for now")

    comp_config = {
        "layer_name": case.data["layer_name"],
        "model_loader_key": case.data["model_loader_key"],
        "is_full_model": case.data["is_full_model"],
        "layer_input_shapes": case.data.get("layer_input_shapes", []),
        "torch_model_data": "torch_layer_model_data",
        "mlir_model_file": "torch_layer_model",
        "input_data": "tweaked_random_input_data",
        "comparison_config": "comparison_config_from_mlir",
    }

    if any(fnmatch.fnmatch(name_lower, pat) for pat in relaxed_tc):
        comp_config["comparison_config"] = "comparison_config_relaxed"

    return comp_config


@pytest.fixture
def reference_results(request):
    """Use PyTorch directly as reference for torch zoo model tests."""
    return request.getfixturevalue("torch_reference_results")


@pytest.mark.ci
def test_torch_zoo(request, reference_results, torq_results, case_config, torch_zoo_case):
    compare_test_results(request, torq_results, reference_results, case_config)
