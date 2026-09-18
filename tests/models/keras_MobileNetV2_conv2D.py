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
def conv2d_test1_int8_mobilenetv2_inp_224x224x3_k3x3_oc32_s2x2_same():
    return conv2d_parametrize((224, 224, 3), 32, (3, 3), (2, 2), "same")


@versioned_unhashable_object_fixture
def conv2d_test2_int8_mobilenetv2_inp_112x112x32_k1x1_oc16_s1x1_same():
    return conv2d_parametrize((112, 112, 32), 16, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test3_int8_mobilenetv2_inp_112x112x16_k1x1_oc96_s1x1_same():
    return conv2d_parametrize((112, 112, 16), 96, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test4_int8_mobilenetv2_inp_56x56x96_k1x1_oc24_s1x1_same():
    return conv2d_parametrize((56, 56, 96), 24, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test5_int8_mobilenetv2_inp_56x56x24_k1x1_oc144_s1x1_same():
    return conv2d_parametrize((56, 56, 24), 144, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test6_int8_mobilenetv2_inp_56x56x144_k1x1_oc24_s1x1_same():
    return conv2d_parametrize((56, 56, 144), 24, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test7_int8_mobilenetv2_inp_56x56x24_k1x1_oc144_s1x1_same():
    return conv2d_parametrize((56, 56, 24), 144, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test8_int8_mobilenetv2_inp_28x28x144_k1x1_oc32_s1x1_same():
    return conv2d_parametrize((28, 28, 144), 32, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test9_int8_mobilenetv2_inp_28x28x32_k1x1_oc192_s1x1_same():
    return conv2d_parametrize((28, 28, 32), 192, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test10_int8_mobilenetv2_inp_28x28x192_k1x1_oc32_s1x1_same():
    return conv2d_parametrize((28, 28, 192), 32, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test11_int8_mobilenetv2_inp_28x28x32_k1x1_oc192_s1x1_same():
    return conv2d_parametrize((28, 28, 32), 192, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test12_int8_mobilenetv2_inp_28x28x192_k1x1_oc32_s1x1_same():
    return conv2d_parametrize((28, 28, 192), 32, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test13_int8_mobilenetv2_inp_28x28x32_k1x1_oc192_s1x1_same():
    return conv2d_parametrize((28, 28, 32), 192, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test14_int8_mobilenetv2_inp_14x14x192_k1x1_oc64_s1x1_same():
    return conv2d_parametrize((14, 14, 192), 64, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test15_int8_mobilenetv2_inp_14x14x64_k1x1_oc384_s1x1_same():
    return conv2d_parametrize((14, 14, 64), 384, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test16_int8_mobilenetv2_inp_14x14x384_k1x1_oc64_s1x1_same():
    return conv2d_parametrize((14, 14, 384), 64, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test17_int8_mobilenetv2_inp_14x14x64_k1x1_oc384_s1x1_same():
    return conv2d_parametrize((14, 14, 64), 384, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test18_int8_mobilenetv2_inp_14x14x384_k1x1_oc64_s1x1_same():
    return conv2d_parametrize((14, 14, 384), 64, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test19_int8_mobilenetv2_inp_14x14x64_k1x1_oc384_s1x1_same():
    return conv2d_parametrize((14, 14, 64), 384, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test20_int8_mobilenetv2_inp_14x14x384_k1x1_oc64_s1x1_same():
    return conv2d_parametrize((14, 14, 384), 64, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test21_int8_mobilenetv2_inp_14x14x64_k1x1_oc384_s1x1_same():
    return conv2d_parametrize((14, 14, 64), 384, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test22_int8_mobilenetv2_inp_14x14x384_k1x1_oc96_s1x1_same():
    return conv2d_parametrize((14, 14, 384), 96, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test23_int8_mobilenetv2_inp_14x14x96_k1x1_oc576_s1x1_same():
    return conv2d_parametrize((14, 14, 96), 576, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test24_int8_mobilenetv2_inp_14x14x576_k1x1_oc96_s1x1_same():
    return conv2d_parametrize((14, 14, 576), 96, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test25_int8_mobilenetv2_inp_14x14x96_k1x1_oc576_s1x1_same():
    return conv2d_parametrize((14, 14, 96), 576, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test26_int8_mobilenetv2_inp_14x14x576_k1x1_oc96_s1x1_same():
    return conv2d_parametrize((14, 14, 576), 96, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test27_int8_mobilenetv2_inp_14x14x96_k1x1_oc576_s1x1_same():
    return conv2d_parametrize((14, 14, 96), 576, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test28_int8_mobilenetv2_inp_7x7x576_k1x1_oc160_s1x1_same():
    return conv2d_parametrize((7, 7, 576), 160, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test29_int8_mobilenetv2_inp_7x7x160_k1x1_oc960_s1x1_same():
    return conv2d_parametrize((7, 7, 160), 960, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test30_int8_mobilenetv2_inp_7x7x960_k1x1_oc160_s1x1_same():
    return conv2d_parametrize((7, 7, 960), 160, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test31_int8_mobilenetv2_inp_7x7x160_k1x1_oc960_s1x1_same():
    return conv2d_parametrize((7, 7, 160), 960, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test32_int8_mobilenetv2_inp_7x7x960_k1x1_oc160_s1x1_same():
    return conv2d_parametrize((7, 7, 960), 160, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test33_int8_mobilenetv2_inp_7x7x160_k1x1_oc960_s1x1_same():
    return conv2d_parametrize((7, 7, 160), 960, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test34_int8_mobilenetv2_inp_7x7x960_k1x1_oc320_s1x1_same():
    return conv2d_parametrize((7, 7, 960), 320, (1, 1), "same")


@versioned_unhashable_object_fixture
def conv2d_test35_int8_mobilenetv2_inp_7x7x320_k1x1_oc1280_s1x1_valid():
    return conv2d_parametrize((7, 7, 320), 1280, (1, 1))


def _is_running_in_ci():
    return any(os.getenv(variable) for variable in (
        "CI",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "JENKINS_HOME",
        "CIRCLECI",
        "BUILD_ID",
    ))


def get_mobilenetv2_conv2d_test_cases():
    test_cases = []

    test_cases.append(Case("conv2d_test1_int8_mobilenetv2_inp_224x224x3_k3x3_oc32_s2x2_same", {
        "keras_model_name": "conv2d_test1_int8_mobilenetv2_inp_224x224x3_k3x3_oc32_s2x2_same"
    }))
    test_cases.append(Case("conv2d_test2_int8_mobilenetv2_inp_112x112x32_k1x1_oc16_s1x1_same", {
        "keras_model_name": "conv2d_test2_int8_mobilenetv2_inp_112x112x32_k1x1_oc16_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test3_int8_mobilenetv2_inp_112x112x16_k1x1_oc96_s1x1_same", {
        "keras_model_name": "conv2d_test3_int8_mobilenetv2_inp_112x112x16_k1x1_oc96_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test4_int8_mobilenetv2_inp_56x56x96_k1x1_oc24_s1x1_same", {
        "keras_model_name": "conv2d_test4_int8_mobilenetv2_inp_56x56x96_k1x1_oc24_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test5_int8_mobilenetv2_inp_56x56x24_k1x1_oc144_s1x1_same", {
        "keras_model_name": "conv2d_test5_int8_mobilenetv2_inp_56x56x24_k1x1_oc144_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test6_int8_mobilenetv2_inp_56x56x144_k1x1_oc24_s1x1_same", {
        "keras_model_name": "conv2d_test6_int8_mobilenetv2_inp_56x56x144_k1x1_oc24_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test7_int8_mobilenetv2_inp_56x56x24_k1x1_oc144_s1x1_same", {
        "keras_model_name": "conv2d_test7_int8_mobilenetv2_inp_56x56x24_k1x1_oc144_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test8_int8_mobilenetv2_inp_28x28x144_k1x1_oc32_s1x1_same", {
        "keras_model_name": "conv2d_test8_int8_mobilenetv2_inp_28x28x144_k1x1_oc32_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test9_int8_mobilenetv2_inp_28x28x32_k1x1_oc192_s1x1_same", {
        "keras_model_name": "conv2d_test9_int8_mobilenetv2_inp_28x28x32_k1x1_oc192_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test10_int8_mobilenetv2_inp_28x28x192_k1x1_oc32_s1x1_same", {
        "keras_model_name": "conv2d_test10_int8_mobilenetv2_inp_28x28x192_k1x1_oc32_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test11_int8_mobilenetv2_inp_28x28x32_k1x1_oc192_s1x1_same", {
        "keras_model_name": "conv2d_test11_int8_mobilenetv2_inp_28x28x32_k1x1_oc192_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test12_int8_mobilenetv2_inp_28x28x192_k1x1_oc32_s1x1_same", {
        "keras_model_name": "conv2d_test12_int8_mobilenetv2_inp_28x28x192_k1x1_oc32_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test13_int8_mobilenetv2_inp_28x28x32_k1x1_oc192_s1x1_same", {
        "keras_model_name": "conv2d_test13_int8_mobilenetv2_inp_28x28x32_k1x1_oc192_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test14_int8_mobilenetv2_inp_14x14x192_k1x1_oc64_s1x1_same", {
        "keras_model_name": "conv2d_test14_int8_mobilenetv2_inp_14x14x192_k1x1_oc64_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test15_int8_mobilenetv2_inp_14x14x64_k1x1_oc384_s1x1_same", {
        "keras_model_name": "conv2d_test15_int8_mobilenetv2_inp_14x14x64_k1x1_oc384_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test16_int8_mobilenetv2_inp_14x14x384_k1x1_oc64_s1x1_same", {
        "keras_model_name": "conv2d_test16_int8_mobilenetv2_inp_14x14x384_k1x1_oc64_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test17_int8_mobilenetv2_inp_14x14x64_k1x1_oc384_s1x1_same", {
        "keras_model_name": "conv2d_test17_int8_mobilenetv2_inp_14x14x64_k1x1_oc384_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test18_int8_mobilenetv2_inp_14x14x384_k1x1_oc64_s1x1_same", {
        "keras_model_name": "conv2d_test18_int8_mobilenetv2_inp_14x14x384_k1x1_oc64_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test19_int8_mobilenetv2_inp_14x14x64_k1x1_oc384_s1x1_same", {
        "keras_model_name": "conv2d_test19_int8_mobilenetv2_inp_14x14x64_k1x1_oc384_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test20_int8_mobilenetv2_inp_14x14x384_k1x1_oc64_s1x1_same", {
        "keras_model_name": "conv2d_test20_int8_mobilenetv2_inp_14x14x384_k1x1_oc64_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test21_int8_mobilenetv2_inp_14x14x64_k1x1_oc384_s1x1_same", {
        "keras_model_name": "conv2d_test21_int8_mobilenetv2_inp_14x14x64_k1x1_oc384_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test22_int8_mobilenetv2_inp_14x14x384_k1x1_oc96_s1x1_same", {
        "keras_model_name": "conv2d_test22_int8_mobilenetv2_inp_14x14x384_k1x1_oc96_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test23_int8_mobilenetv2_inp_14x14x96_k1x1_oc576_s1x1_same", {
        "keras_model_name": "conv2d_test23_int8_mobilenetv2_inp_14x14x96_k1x1_oc576_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test24_int8_mobilenetv2_inp_14x14x576_k1x1_oc96_s1x1_same", {
        "keras_model_name": "conv2d_test24_int8_mobilenetv2_inp_14x14x576_k1x1_oc96_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test25_int8_mobilenetv2_inp_14x14x96_k1x1_oc576_s1x1_same", {
        "keras_model_name": "conv2d_test25_int8_mobilenetv2_inp_14x14x96_k1x1_oc576_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test26_int8_mobilenetv2_inp_14x14x576_k1x1_oc96_s1x1_same", {
        "keras_model_name": "conv2d_test26_int8_mobilenetv2_inp_14x14x576_k1x1_oc96_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test27_int8_mobilenetv2_inp_14x14x96_k1x1_oc576_s1x1_same", {
        "keras_model_name": "conv2d_test27_int8_mobilenetv2_inp_14x14x96_k1x1_oc576_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test28_int8_mobilenetv2_inp_7x7x576_k1x1_oc160_s1x1_same", {
        "keras_model_name": "conv2d_test28_int8_mobilenetv2_inp_7x7x576_k1x1_oc160_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test29_int8_mobilenetv2_inp_7x7x160_k1x1_oc960_s1x1_same", {
        "keras_model_name": "conv2d_test29_int8_mobilenetv2_inp_7x7x160_k1x1_oc960_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test30_int8_mobilenetv2_inp_7x7x960_k1x1_oc160_s1x1_same", {
        "keras_model_name": "conv2d_test30_int8_mobilenetv2_inp_7x7x960_k1x1_oc160_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test31_int8_mobilenetv2_inp_7x7x160_k1x1_oc960_s1x1_same", {
        "keras_model_name": "conv2d_test31_int8_mobilenetv2_inp_7x7x160_k1x1_oc960_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test32_int8_mobilenetv2_inp_7x7x960_k1x1_oc160_s1x1_same", {
        "keras_model_name": "conv2d_test32_int8_mobilenetv2_inp_7x7x960_k1x1_oc160_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test33_int8_mobilenetv2_inp_7x7x160_k1x1_oc960_s1x1_same", {
        "keras_model_name": "conv2d_test33_int8_mobilenetv2_inp_7x7x160_k1x1_oc960_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test34_int8_mobilenetv2_inp_7x7x960_k1x1_oc320_s1x1_same", {
        "keras_model_name": "conv2d_test34_int8_mobilenetv2_inp_7x7x960_k1x1_oc320_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test35_int8_mobilenetv2_inp_7x7x320_k1x1_oc1280_s1x1_valid", {
        "keras_model_name": "conv2d_test35_int8_mobilenetv2_inp_7x7x320_k1x1_oc1280_s1x1_valid"
    }))

    if _is_running_in_ci():
        ci_case_names = {
            "conv2d_test1_int8_mobilenetv2_inp_224x224x3_k3x3_oc32_s2x2_same",
            "conv2d_test3_int8_mobilenetv2_inp_112x112x16_k1x1_oc96_s1x1_same",
        }
        return [case for case in test_cases if case.name in ci_case_names]

    return test_cases
