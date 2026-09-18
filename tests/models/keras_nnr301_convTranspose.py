import tensorflow as tf

from torq.testing.versioned_fixtures import versioned_unhashable_object_fixture
from torq.testing.cases import Case


def convtranspose_parametrize(
    shape,
    filters,
    kernel_size,
    strides=(1, 1),
    padding="valid",
    **convtranspose_kwargs,
):
    tf.keras.utils.set_random_seed(42)
    if isinstance(strides, str):
        padding = strides
        strides = (1, 1)

    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=shape),
        tf.keras.layers.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            use_bias=True,
            **convtranspose_kwargs,
        ),
    ])


@versioned_unhashable_object_fixture
def convtranspose_test1_int8_nnr301_inp_16x1x48_k3x1_oc48_s2x1_same():
    return convtranspose_parametrize((16, 1, 48), 48, (3, 1), (2, 1), "same")


@versioned_unhashable_object_fixture
def convtranspose_test2_int8_nnr301_inp_32x1x48_k3x1_oc48_s2x1_same():
    return convtranspose_parametrize((32, 1, 48), 48, (3, 1), (2, 1), "same")


@versioned_unhashable_object_fixture
def convtranspose_test3_int8_nnr301_inp_64x1x32_k3x1_oc32_s1x1_same():
    return convtranspose_parametrize((64, 1, 32), 32, (3, 1), "same")


@versioned_unhashable_object_fixture
def convtranspose_test4_int8_nnr301_inp_64x1x32_k3x1_oc32_s2x1_same():
    return convtranspose_parametrize((64, 1, 32), 32, (3, 1), (2, 1), "same")


@versioned_unhashable_object_fixture
def convtranspose_test5_int8_nnr301_inp_128x1x32_k3x1_oc32_s1x1_same():
    return convtranspose_parametrize((128, 1, 32), 32, (3, 1), "same")


@versioned_unhashable_object_fixture
def convtranspose_test6_int8_nnr301_inp_128x1x1_k3x1_oc1_s2x1_same():
    return convtranspose_parametrize((128, 1, 1), 1, (3, 1), (2, 1), "same")


def get_nnr301_convtranspose_test_cases():
    test_cases = []

    test_cases.append(Case("convtranspose_test1_int8_nnr301_inp_16x1x48_k3x1_oc48_s2x1_same", {
        "keras_model_name": "convtranspose_test1_int8_nnr301_inp_16x1x48_k3x1_oc48_s2x1_same"
    }))
    test_cases.append(Case("convtranspose_test2_int8_nnr301_inp_32x1x48_k3x1_oc48_s2x1_same", {
        "keras_model_name": "convtranspose_test2_int8_nnr301_inp_32x1x48_k3x1_oc48_s2x1_same"
    }))
    test_cases.append(Case("convtranspose_test3_int8_nnr301_inp_64x1x32_k3x1_oc32_s1x1_same", {
        "keras_model_name": "convtranspose_test3_int8_nnr301_inp_64x1x32_k3x1_oc32_s1x1_same"
    }))
    test_cases.append(Case("convtranspose_test4_int8_nnr301_inp_64x1x32_k3x1_oc32_s2x1_same", {
        "keras_model_name": "convtranspose_test4_int8_nnr301_inp_64x1x32_k3x1_oc32_s2x1_same"
    }))
    test_cases.append(Case("convtranspose_test5_int8_nnr301_inp_128x1x32_k3x1_oc32_s1x1_same", {
        "keras_model_name": "convtranspose_test5_int8_nnr301_inp_128x1x32_k3x1_oc32_s1x1_same"
    }))
    test_cases.append(Case("convtranspose_test6_int8_nnr301_inp_128x1x1_k3x1_oc1_s2x1_same", {
        "keras_model_name": "convtranspose_test6_int8_nnr301_inp_128x1x1_k3x1_oc1_s2x1_same"
    }))

    return test_cases
