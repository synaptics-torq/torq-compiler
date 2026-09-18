import os

import tensorflow as tf

from torq.testing.cases import Case
from torq.testing.versioned_fixtures import versioned_unhashable_object_fixture


def conv2d_parametrize(
    shape,
    filters,
    kernel_size,
    **conv2d_kwargs,
):
    tf.keras.utils.set_random_seed(42)

    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=shape),
        tf.keras.layers.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            **conv2d_kwargs,
        ),
    ])


@versioned_unhashable_object_fixture
def conv2d_test1_int8_inp_136x160x1_k3x3_oc5_s1x1_valid():
    return conv2d_parametrize((136, 160, 1), 5, (3, 3))


@versioned_unhashable_object_fixture
def conv2d_test2_int8_inp_67x79x5_k3x3_oc10_s1x1_valid():
    return conv2d_parametrize((67, 79, 5), 10, (3, 3))


@versioned_unhashable_object_fixture
def conv2d_test3_int8_inp_30x36x10_k1x1_oc15_s1x1_valid():
    return conv2d_parametrize((30, 36, 10), 15, (1, 1))


@versioned_unhashable_object_fixture
def conv2d_test4_int8_inp_15x18x15_k4x3_oc64_s1x1_valid():
    return conv2d_parametrize((15, 18, 15), 64, (4, 3))


@versioned_unhashable_object_fixture
def conv2d_test5_int8_inp_12x16x64_k1x1_oc32_s1x1_valid():
    return conv2d_parametrize((12, 16, 64), 32, (1, 1))


@versioned_unhashable_object_fixture
def conv2d_test6_int8_inp_12x16x32_k1x1_oc4_s1x1_valid():
    return conv2d_parametrize((12, 16, 32), 4, (1, 1))


@versioned_unhashable_object_fixture
def conv2d_test7_int8_inp_12x16x32_k1x1_oc1_s1x1_valid():
    return conv2d_parametrize((12, 16, 32), 1, (1, 1))


def _is_running_in_ci():
    return any(os.getenv(variable) for variable in (
        "CI",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "JENKINS_HOME",
        "CIRCLECI",
        "BUILD_ID",
    ))


def get_emza70_conv2d_test_cases():
    all_cases = []

    all_cases.append(Case("conv2d_test1_int8_inp_136x160x1_k3x3_oc5_s1x1_valid", {
        "keras_model_name": "conv2d_test1_int8_inp_136x160x1_k3x3_oc5_s1x1_valid"
    }))
    all_cases.append(Case("conv2d_test2_int8_inp_67x79x5_k3x3_oc10_s1x1_valid", {
        "keras_model_name": "conv2d_test2_int8_inp_67x79x5_k3x3_oc10_s1x1_valid"
    }))
    all_cases.append(Case("conv2d_test3_int8_inp_30x36x10_k1x1_oc15_s1x1_valid", {
        "keras_model_name": "conv2d_test3_int8_inp_30x36x10_k1x1_oc15_s1x1_valid"
    }))
    all_cases.append(Case("conv2d_test4_int8_inp_15x18x15_k4x3_oc64_s1x1_valid", {
        "keras_model_name": "conv2d_test4_int8_inp_15x18x15_k4x3_oc64_s1x1_valid"
    }))
    all_cases.append(Case("conv2d_test5_int8_inp_12x16x64_k1x1_oc32_s1x1_valid", {
        "keras_model_name": "conv2d_test5_int8_inp_12x16x64_k1x1_oc32_s1x1_valid"
    }))
    all_cases.append(Case("conv2d_test6_int8_inp_12x16x32_k1x1_oc4_s1x1_valid", {
        "keras_model_name": "conv2d_test6_int8_inp_12x16x32_k1x1_oc4_s1x1_valid"
    }))
    all_cases.append(Case("conv2d_test7_int8_inp_12x16x32_k1x1_oc1_s1x1_valid", {
        "keras_model_name": "conv2d_test7_int8_inp_12x16x32_k1x1_oc1_s1x1_valid"
    }))

    if _is_running_in_ci():
        ci_case_names = {
            "conv2d_test3_int8_inp_30x36x10_k1x1_oc15_s1x1_valid",
        }
        return [case for case in all_cases if case.name in ci_case_names]

    return all_cases
