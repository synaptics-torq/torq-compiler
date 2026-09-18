import os

import tensorflow as tf

from torq.testing.versioned_fixtures import versioned_unhashable_object_fixture
from torq.testing.cases import Case


def conv2d_parametrize(
    shape,
    filters,
    kernel_size,
    strides=(1, 1),
    padding="valid",
    **conv2d_kwargs,
):
    tf.keras.utils.set_random_seed(42)
    if isinstance(strides, str):
        padding = strides
        strides = (1, 1)

    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=shape),
        tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=True,
            **conv2d_kwargs,
        ),
    ])


@versioned_unhashable_object_fixture
def conv2d_test1_int8_inp_256x1x1_k3x1_oc32_s2x1_same():
    return conv2d_parametrize((256, 1, 1), 32, (3, 1), (2, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test2_int8_inp_128x1x32_k1x1_oc32_s1x1_valid():
    return conv2d_parametrize((128, 1, 32), 32, (1, 1))


@versioned_unhashable_object_fixture
def conv2d_test3_int8_inp_64x1x32_k1x1_oc64_s1x1_valid():
    return conv2d_parametrize((64, 1, 32), 64, (1, 1))


@versioned_unhashable_object_fixture
def conv2d_test4_int8_inp_64x1x64_k1x1_oc64_s1x1_valid():
    return conv2d_parametrize((64, 1, 64), 64, (1, 1))


@versioned_unhashable_object_fixture
def conv2d_test5_int8_inp_32x1x64_k1x1_oc64_s1x1_valid():
    return conv2d_parametrize((32, 1, 64), 64, (1, 1))


@versioned_unhashable_object_fixture
def conv2d_test6_int8_inp_16x1x64_k1x1_oc64_s1x1_valid():
    return conv2d_parametrize((16, 1, 64), 64, (1, 1))


@versioned_unhashable_object_fixture
def conv2d_test7_int8_inp_16x1x64_k1x1_oc32_s1x1_same():
    return conv2d_parametrize((16, 1, 64), 32, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test8_int8_inp_16x1x32_k1x1_oc48_s1x1_valid():
    return conv2d_parametrize((16, 1, 32), 48, (1, 1))


@versioned_unhashable_object_fixture
def conv2d_test9_int8_inp_16x1x48_k2x1_oc48_s1x1_same():
    return conv2d_parametrize((16, 1, 48), 48, (2, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test10_int8_inp_16x1x48_k1x1_oc48_s1x1_valid():
    return conv2d_parametrize((16, 1, 48), 48, (1, 1))


@versioned_unhashable_object_fixture
def conv2d_test11_int8_inp_32x1x48_k1x1_oc48_s1x1_valid():
    return conv2d_parametrize((32, 1, 48), 48, (1, 1))


@versioned_unhashable_object_fixture
def conv2d_test12_int8_inp_32x1x48_k2x1_oc48_s1x1_same():
    return conv2d_parametrize((32, 1, 48), 48, (2, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test13_int8_inp_64x1x48_k1x1_oc32_s1x1_valid():
    return conv2d_parametrize((64, 1, 48), 32, (1, 1))


@versioned_unhashable_object_fixture
def conv2d_test14_int8_inp_64x1x32_k3x1_oc32_s1x1_same():
    return conv2d_parametrize((64, 1, 32), 32, (3, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test15_int8_inp_64x1x32_k1x1_oc32_s1x1_valid():
    return conv2d_parametrize((64, 1, 32), 32, (1, 1))


@versioned_unhashable_object_fixture
def conv2d_test16_int8_inp_64x1x32_k2x1_oc32_s1x1_same():
    return conv2d_parametrize((64, 1, 32), 32, (2, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test17_int8_inp_128x1x32_k3x1_oc32_s1x1_same():
    return conv2d_parametrize((128, 1, 32), 32, (3, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test18_int8_inp_128x1x32_k1x1_oc1_s1x1_valid():
    return conv2d_parametrize((128, 1, 32), 1, (1, 1))


@versioned_unhashable_object_fixture
def conv2d_test19_int8_inp_128x1x1_k2x1_oc1_s1x1_same():
    return conv2d_parametrize((128, 1, 1), 1, (2, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test20_int8_inp_128x1x1_k1x1_oc1_s1x1_valid():
    return conv2d_parametrize((128, 1, 1), 1, (1, 1))


def _is_running_in_ci():
    return any(os.getenv(variable) for variable in (
        "CI",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "JENKINS_HOME",
        "CIRCLECI",
        "BUILD_ID",
    ))


def get_nnr301_conv2d_test_cases():
    test_cases = []

    test_cases.append(Case("conv2d_test1_int8_inp_256x1x1_k3x1_oc32_s2x1_same", {
        "keras_model_name": "conv2d_test1_int8_inp_256x1x1_k3x1_oc32_s2x1_same"
    }))
    test_cases.append(Case("conv2d_test2_int8_inp_128x1x32_k1x1_oc32_s1x1_valid", {
        "keras_model_name": "conv2d_test2_int8_inp_128x1x32_k1x1_oc32_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test3_int8_inp_64x1x32_k1x1_oc64_s1x1_valid", {
        "keras_model_name": "conv2d_test3_int8_inp_64x1x32_k1x1_oc64_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test4_int8_inp_64x1x64_k1x1_oc64_s1x1_valid", {
        "keras_model_name": "conv2d_test4_int8_inp_64x1x64_k1x1_oc64_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test5_int8_inp_32x1x64_k1x1_oc64_s1x1_valid", {
        "keras_model_name": "conv2d_test5_int8_inp_32x1x64_k1x1_oc64_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test6_int8_inp_16x1x64_k1x1_oc64_s1x1_valid", {
        "keras_model_name": "conv2d_test6_int8_inp_16x1x64_k1x1_oc64_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test7_int8_inp_16x1x64_k1x1_oc32_s1x1_same", {
        "keras_model_name": "conv2d_test7_int8_inp_16x1x64_k1x1_oc32_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test8_int8_inp_16x1x32_k1x1_oc48_s1x1_valid", {
        "keras_model_name": "conv2d_test8_int8_inp_16x1x32_k1x1_oc48_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test9_int8_inp_16x1x48_k2x1_oc48_s1x1_same", {
        "keras_model_name": "conv2d_test9_int8_inp_16x1x48_k2x1_oc48_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test10_int8_inp_16x1x48_k1x1_oc48_s1x1_valid", {
        "keras_model_name": "conv2d_test10_int8_inp_16x1x48_k1x1_oc48_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test11_int8_inp_32x1x48_k1x1_oc48_s1x1_valid", {
        "keras_model_name": "conv2d_test11_int8_inp_32x1x48_k1x1_oc48_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test12_int8_inp_32x1x48_k2x1_oc48_s1x1_same", {
        "keras_model_name": "conv2d_test12_int8_inp_32x1x48_k2x1_oc48_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test13_int8_inp_64x1x48_k1x1_oc32_s1x1_valid", {
        "keras_model_name": "conv2d_test13_int8_inp_64x1x48_k1x1_oc32_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test14_int8_inp_64x1x32_k3x1_oc32_s1x1_same", {
        "keras_model_name": "conv2d_test14_int8_inp_64x1x32_k3x1_oc32_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test15_int8_inp_64x1x32_k1x1_oc32_s1x1_valid", {
        "keras_model_name": "conv2d_test15_int8_inp_64x1x32_k1x1_oc32_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test16_int8_inp_64x1x32_k2x1_oc32_s1x1_same", {
        "keras_model_name": "conv2d_test16_int8_inp_64x1x32_k2x1_oc32_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test17_int8_inp_128x1x32_k3x1_oc32_s1x1_same", {
        "keras_model_name": "conv2d_test17_int8_inp_128x1x32_k3x1_oc32_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test18_int8_inp_128x1x32_k1x1_oc1_s1x1_valid", {
        "keras_model_name": "conv2d_test18_int8_inp_128x1x32_k1x1_oc1_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test19_int8_inp_128x1x1_k2x1_oc1_s1x1_same", {
        "keras_model_name": "conv2d_test19_int8_inp_128x1x1_k2x1_oc1_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test20_int8_inp_128x1x1_k1x1_oc1_s1x1_valid", {
        "keras_model_name": "conv2d_test20_int8_inp_128x1x1_k1x1_oc1_s1x1_valid"
    }))

    if _is_running_in_ci():
        ci_case_names = {
            "conv2d_test1_int8_inp_256x1x1_k3x1_oc32_s2x1_same",
            "conv2d_test6_int8_inp_16x1x64_k1x1_oc64_s1x1_valid",
            "conv2d_test7_int8_inp_16x1x64_k1x1_oc32_s1x1_same",
            "conv2d_test8_int8_inp_16x1x32_k1x1_oc48_s1x1_valid",
            "conv2d_test9_int8_inp_16x1x48_k2x1_oc48_s1x1_same",
            "conv2d_test10_int8_inp_16x1x48_k1x1_oc48_s1x1_valid",
            "conv2d_test18_int8_inp_128x1x32_k1x1_oc1_s1x1_valid",
            "conv2d_test19_int8_inp_128x1x1_k2x1_oc1_s1x1_same",
            "conv2d_test20_int8_inp_128x1x1_k1x1_oc1_s1x1_valid",
        }
        return [case for case in test_cases if case.name in ci_case_names]

    return test_cases