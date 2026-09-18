import os

import tensorflow as tf

from torq.testing.versioned_fixtures import versioned_unhashable_object_fixture
from torq.testing.cases import Case


def depthwise_parametrize(
    shape,
    kernel_size,
    strides=(1, 1),
    padding="valid",
    **depthwise_kwargs,
):
    tf.keras.utils.set_random_seed(42)
    if isinstance(strides, str):
        padding = strides
        strides = (1, 1)

    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=shape),
        tf.keras.layers.DepthwiseConv2D(
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            depth_multiplier=1,
            use_bias=True,
            **depthwise_kwargs,
        ),
    ])


# --- EMZA_70 ---

@versioned_unhashable_object_fixture
def dw001_EMZA_70_dw_in38x32x10_k3x3_dm1_s1x1_valid():
    return depthwise_parametrize((38, 32, 10), (3, 3))


# --- EMZA_75 ---

@versioned_unhashable_object_fixture
def dw002_EMZA_75_dw_in122x70x16_k3x3_dm1_s1x1_valid():
    return depthwise_parametrize((122, 70, 16), (3, 3))


@versioned_unhashable_object_fixture
def dw003_EMZA_75_dw_in122x70x32_k3x3_dm1_s2x2_valid():
    return depthwise_parametrize((122, 70, 32), (3, 3), (2, 2))


@versioned_unhashable_object_fixture
def dw004_EMZA_75_dw_in62x36x32_k3x3_dm1_s1x1_valid():
    return depthwise_parametrize((62, 36, 32), (3, 3))


@versioned_unhashable_object_fixture
def dw005_EMZA_75_dw_in62x36x32_k3x3_dm1_s2x2_valid():
    return depthwise_parametrize((62, 36, 32), (3, 3), (2, 2))


@versioned_unhashable_object_fixture
def dw006_EMZA_75_dw_in32x19x64_k3x3_dm1_s1x1_valid():
    return depthwise_parametrize((32, 19, 64), (3, 3))


@versioned_unhashable_object_fixture
def dw007_EMZA_75_dw_in32x19x64_k3x3_dm1_s2x2_valid():
    return depthwise_parametrize((32, 19, 64), (3, 3), (2, 2))


@versioned_unhashable_object_fixture
def dw008_EMZA_75_dw_in17x11x128_k3x3_dm1_s1x1_valid():
    return depthwise_parametrize((17, 11, 128), (3, 3))


@versioned_unhashable_object_fixture
def dw009_EMZA_75_dw_in17x11x128_k3x3_dm1_s2x2_valid():
    return depthwise_parametrize((17, 11, 128), (3, 3), (2, 2))


@versioned_unhashable_object_fixture
def dw010_EMZA_75_dw_in10x7x256_k3x3_dm1_s1x1_valid():
    return depthwise_parametrize((10, 7, 256), (3, 3))


@versioned_unhashable_object_fixture
def dw011_EMZA_75_dw_in10x7x64_k3x3_dm1_s2x2_valid():
    return depthwise_parametrize((10, 7, 64), (3, 3), (2, 2))


# --- MobileNetV2 ---

@versioned_unhashable_object_fixture
def dw012_MobileNetV2_dw_in112x112x32_k3x3_dm1_s1x1_same():
    return depthwise_parametrize((112, 112, 32), (3, 3), "same")


@versioned_unhashable_object_fixture
def dw013_MobileNetV2_dw_in112x112x96_k3x3_dm1_s2x2_same():
    return depthwise_parametrize((112, 112, 96), (3, 3), (2, 2), "same")


@versioned_unhashable_object_fixture
def dw014_MobileNetV2_dw_in56x56x144_k3x3_dm1_s1x1_same():
    return depthwise_parametrize((56, 56, 144), (3, 3), "same")


@versioned_unhashable_object_fixture
def dw015_MobileNetV2_dw_in56x56x144_k3x3_dm1_s2x2_same():
    return depthwise_parametrize((56, 56, 144), (3, 3), (2, 2), "same")


@versioned_unhashable_object_fixture
def dw016_MobileNetV2_dw_in28x28x192_k3x3_dm1_s1x1_same():
    return depthwise_parametrize((28, 28, 192), (3, 3), "same")


@versioned_unhashable_object_fixture
def dw017_MobileNetV2_dw_in28x28x192_k3x3_dm1_s2x2_same():
    return depthwise_parametrize((28, 28, 192), (3, 3), (2, 2), "same")


@versioned_unhashable_object_fixture
def dw018_MobileNetV2_dw_in14x14x384_k3x3_dm1_s1x1_same():
    return depthwise_parametrize((14, 14, 384), (3, 3), "same")


@versioned_unhashable_object_fixture
def dw019_MobileNetV2_dw_in14x14x576_k3x3_dm1_s1x1_same():
    return depthwise_parametrize((14, 14, 576), (3, 3), "same")


@versioned_unhashable_object_fixture
def dw020_MobileNetV2_dw_in14x14x576_k3x3_dm1_s2x2_same():
    return depthwise_parametrize((14, 14, 576), (3, 3), (2, 2), "same")


@versioned_unhashable_object_fixture
def dw021_MobileNetV2_dw_in7x7x960_k3x3_dm1_s1x1_same():
    return depthwise_parametrize((7, 7, 960), (3, 3), "same")


# --- NNR_301 ---

@versioned_unhashable_object_fixture
def dw022_NNR_301_dw_in128x1x32_k1x3_dm1_s1x1_same():
    return depthwise_parametrize((128, 1, 32), (1, 3), "same")


@versioned_unhashable_object_fixture
def dw023_NNR_301_dw_in128x1x32_k1x3_dm1_s2x2_same():
    return depthwise_parametrize((128, 1, 32), (1, 3), (1, 2), "same")


@versioned_unhashable_object_fixture
def dw024_NNR_301_dw_in64x1x64_k1x3_dm1_s1x1_same():
    return depthwise_parametrize((64, 1, 64), (1, 3), "same")


@versioned_unhashable_object_fixture
def dw025_NNR_301_dw_in64x1x64_k1x3_dm1_s2x2_same():
    return depthwise_parametrize((64, 1, 64), (1, 3), (1, 2), "same")


@versioned_unhashable_object_fixture
def dw026_NNR_301_dw_in32x1x64_k1x3_dm1_s2x2_same():
    return depthwise_parametrize((32, 1, 64), (1, 3), (1, 2), "same")


# --- MNIST ---

@versioned_unhashable_object_fixture
def dw027_mnist_dw_in26x26x8_k3x3_dm1_s2x2_same():
    return depthwise_parametrize((26, 26, 8), (3, 3), (2, 2), "same")


def _is_running_in_ci():
    return any(os.getenv(variable) for variable in (
        "CI",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "JENKINS_HOME",
        "CIRCLECI",
        "BUILD_ID",
    ))


def get_audio_vision_depthwise_test_cases():
    test_cases = []

    # Add dw001_EMZA_70_dw_in38x32x10_k3x3_dm1_s1x1_valid test case
    test_cases.append(Case("dw001_EMZA_70_dw_in38x32x10_k3x3_dm1_s1x1_valid", {
        "keras_model_name": "dw001_EMZA_70_dw_in38x32x10_k3x3_dm1_s1x1_valid"
    }))

    # Add dw002_EMZA_75_dw_in122x70x16_k3x3_dm1_s1x1_valid test case
    test_cases.append(Case("dw002_EMZA_75_dw_in122x70x16_k3x3_dm1_s1x1_valid", {
        "keras_model_name": "dw002_EMZA_75_dw_in122x70x16_k3x3_dm1_s1x1_valid"
    }))

    # Add dw003_EMZA_75_dw_in122x70x32_k3x3_dm1_s2x2_valid test case
    test_cases.append(Case("dw003_EMZA_75_dw_in122x70x32_k3x3_dm1_s2x2_valid", {
        "keras_model_name": "dw003_EMZA_75_dw_in122x70x32_k3x3_dm1_s2x2_valid"
    }))

    # Add dw004_EMZA_75_dw_in62x36x32_k3x3_dm1_s1x1_valid test case
    test_cases.append(Case("dw004_EMZA_75_dw_in62x36x32_k3x3_dm1_s1x1_valid", {
        "keras_model_name": "dw004_EMZA_75_dw_in62x36x32_k3x3_dm1_s1x1_valid"
    }))

    # Add dw005_EMZA_75_dw_in62x36x32_k3x3_dm1_s2x2_valid test case
    test_cases.append(Case("dw005_EMZA_75_dw_in62x36x32_k3x3_dm1_s2x2_valid", {
        "keras_model_name": "dw005_EMZA_75_dw_in62x36x32_k3x3_dm1_s2x2_valid"
    }))

    # Add dw006_EMZA_75_dw_in32x19x64_k3x3_dm1_s1x1_valid test case
    test_cases.append(Case("dw006_EMZA_75_dw_in32x19x64_k3x3_dm1_s1x1_valid", {
        "keras_model_name": "dw006_EMZA_75_dw_in32x19x64_k3x3_dm1_s1x1_valid"
    }))

    # Add dw007_EMZA_75_dw_in32x19x64_k3x3_dm1_s2x2_valid test case
    test_cases.append(Case("dw007_EMZA_75_dw_in32x19x64_k3x3_dm1_s2x2_valid", {
        "keras_model_name": "dw007_EMZA_75_dw_in32x19x64_k3x3_dm1_s2x2_valid"
    }))

    # Add dw008_EMZA_75_dw_in17x11x128_k3x3_dm1_s1x1_valid test case
    test_cases.append(Case("dw008_EMZA_75_dw_in17x11x128_k3x3_dm1_s1x1_valid", {
        "keras_model_name": "dw008_EMZA_75_dw_in17x11x128_k3x3_dm1_s1x1_valid"
    }))

    # Add dw009_EMZA_75_dw_in17x11x128_k3x3_dm1_s2x2_valid test case
    test_cases.append(Case("dw009_EMZA_75_dw_in17x11x128_k3x3_dm1_s2x2_valid", {
        "keras_model_name": "dw009_EMZA_75_dw_in17x11x128_k3x3_dm1_s2x2_valid"
    }))

    # Add dw010_EMZA_75_dw_in10x7x256_k3x3_dm1_s1x1_valid test case
    test_cases.append(Case("dw010_EMZA_75_dw_in10x7x256_k3x3_dm1_s1x1_valid", {
        "keras_model_name": "dw010_EMZA_75_dw_in10x7x256_k3x3_dm1_s1x1_valid"
    }))

    # Add dw011_EMZA_75_dw_in10x7x64_k3x3_dm1_s2x2_valid test case
    test_cases.append(Case("dw011_EMZA_75_dw_in10x7x64_k3x3_dm1_s2x2_valid", {
        "keras_model_name": "dw011_EMZA_75_dw_in10x7x64_k3x3_dm1_s2x2_valid"
    }))

    # Add dw012_MobileNetV2_dw_in112x112x32_k3x3_dm1_s1x1_same test case
    test_cases.append(Case("dw012_MobileNetV2_dw_in112x112x32_k3x3_dm1_s1x1_same", {
        "keras_model_name": "dw012_MobileNetV2_dw_in112x112x32_k3x3_dm1_s1x1_same"
    }))

    # Add dw013_MobileNetV2_dw_in112x112x96_k3x3_dm1_s2x2_same test case
    test_cases.append(Case("dw013_MobileNetV2_dw_in112x112x96_k3x3_dm1_s2x2_same", {
        "keras_model_name": "dw013_MobileNetV2_dw_in112x112x96_k3x3_dm1_s2x2_same"
    }))

    # Add dw014_MobileNetV2_dw_in56x56x144_k3x3_dm1_s1x1_same test case
    test_cases.append(Case("dw014_MobileNetV2_dw_in56x56x144_k3x3_dm1_s1x1_same", {
        "keras_model_name": "dw014_MobileNetV2_dw_in56x56x144_k3x3_dm1_s1x1_same"
    }))

    # Add dw015_MobileNetV2_dw_in56x56x144_k3x3_dm1_s2x2_same test case
    test_cases.append(Case("dw015_MobileNetV2_dw_in56x56x144_k3x3_dm1_s2x2_same", {
        "keras_model_name": "dw015_MobileNetV2_dw_in56x56x144_k3x3_dm1_s2x2_same"
    }))

    # Add dw016_MobileNetV2_dw_in28x28x192_k3x3_dm1_s1x1_same test case
    test_cases.append(Case("dw016_MobileNetV2_dw_in28x28x192_k3x3_dm1_s1x1_same", {
        "keras_model_name": "dw016_MobileNetV2_dw_in28x28x192_k3x3_dm1_s1x1_same"
    }))

    # Add dw017_MobileNetV2_dw_in28x28x192_k3x3_dm1_s2x2_same test case
    test_cases.append(Case("dw017_MobileNetV2_dw_in28x28x192_k3x3_dm1_s2x2_same", {
        "keras_model_name": "dw017_MobileNetV2_dw_in28x28x192_k3x3_dm1_s2x2_same"
    }))

    # Add dw018_MobileNetV2_dw_in14x14x384_k3x3_dm1_s1x1_same test case
    test_cases.append(Case("dw018_MobileNetV2_dw_in14x14x384_k3x3_dm1_s1x1_same", {
        "keras_model_name": "dw018_MobileNetV2_dw_in14x14x384_k3x3_dm1_s1x1_same"
    }))

    # Add dw019_MobileNetV2_dw_in14x14x576_k3x3_dm1_s1x1_same test case
    test_cases.append(Case("dw019_MobileNetV2_dw_in14x14x576_k3x3_dm1_s1x1_same", {
        "keras_model_name": "dw019_MobileNetV2_dw_in14x14x576_k3x3_dm1_s1x1_same"
    }))

    # Add dw020_MobileNetV2_dw_in14x14x576_k3x3_dm1_s2x2_same test case
    test_cases.append(Case("dw020_MobileNetV2_dw_in14x14x576_k3x3_dm1_s2x2_same", {
        "keras_model_name": "dw020_MobileNetV2_dw_in14x14x576_k3x3_dm1_s2x2_same"
    }))

    # Add dw021_MobileNetV2_dw_in7x7x960_k3x3_dm1_s1x1_same test case
    test_cases.append(Case("dw021_MobileNetV2_dw_in7x7x960_k3x3_dm1_s1x1_same", {
        "keras_model_name": "dw021_MobileNetV2_dw_in7x7x960_k3x3_dm1_s1x1_same"
    }))

    # Add dw022_NNR_301_dw_in128x1x32_k1x3_dm1_s1x1_same test case
    test_cases.append(Case("dw022_NNR_301_dw_in128x1x32_k1x3_dm1_s1x1_same", {
        "keras_model_name": "dw022_NNR_301_dw_in128x1x32_k1x3_dm1_s1x1_same"
    }))

    # Add dw023_NNR_301_dw_in128x1x32_k1x3_dm1_s2x2_same test case
    test_cases.append(Case("dw023_NNR_301_dw_in128x1x32_k1x3_dm1_s2x2_same", {
        "keras_model_name": "dw023_NNR_301_dw_in128x1x32_k1x3_dm1_s2x2_same"
    }))

    # Add dw024_NNR_301_dw_in64x1x64_k1x3_dm1_s1x1_same test case
    test_cases.append(Case("dw024_NNR_301_dw_in64x1x64_k1x3_dm1_s1x1_same", {
        "keras_model_name": "dw024_NNR_301_dw_in64x1x64_k1x3_dm1_s1x1_same"
    }))

    # Add dw025_NNR_301_dw_in64x1x64_k1x3_dm1_s2x2_same test case
    test_cases.append(Case("dw025_NNR_301_dw_in64x1x64_k1x3_dm1_s2x2_same", {
        "keras_model_name": "dw025_NNR_301_dw_in64x1x64_k1x3_dm1_s2x2_same"
    }))

    # Add dw026_NNR_301_dw_in32x1x64_k1x3_dm1_s2x2_same test case
    test_cases.append(Case("dw026_NNR_301_dw_in32x1x64_k1x3_dm1_s2x2_same", {
        "keras_model_name": "dw026_NNR_301_dw_in32x1x64_k1x3_dm1_s2x2_same"
    }))

    # Add dw027_mnist_dw_in26x26x8_k3x3_dm1_s2x2_same test case
    test_cases.append(Case("dw027_mnist_dw_in26x26x8_k3x3_dm1_s2x2_same", {
        "keras_model_name": "dw027_mnist_dw_in26x26x8_k3x3_dm1_s2x2_same"
    }))

    if _is_running_in_ci():
        ci_case_names = {
            "dw003_EMZA_75_dw_in122x70x32_k3x3_dm1_s2x2_valid",
            "dw011_EMZA_75_dw_in10x7x64_k3x3_dm1_s2x2_valid",
            "dw013_MobileNetV2_dw_in112x112x96_k3x3_dm1_s2x2_same",
            "dw022_NNR_301_dw_in128x1x32_k1x3_dm1_s1x1_same",
            "dw023_NNR_301_dw_in128x1x32_k1x3_dm1_s2x2_same",
            "dw026_NNR_301_dw_in32x1x64_k1x3_dm1_s2x2_same",
            "dw027_mnist_dw_in26x26x8_k3x3_dm1_s2x2_same",
        }
        return [case for case in test_cases if case.name in ci_case_names]

    return test_cases
