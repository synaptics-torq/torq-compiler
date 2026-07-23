import tensorflow as tf

from torq.testing.versioned_fixtures import versioned_unhashable_object_fixture
from torq.testing.cases import Case


@versioned_unhashable_object_fixture
def convtranspose_test1_int8_nnr301_inp_16x1x48_k3x1_oc48_s2x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(16, 1, 48)),
        tf.keras.layers.Conv2DTranspose(
            filters=48,
            kernel_size=(3, 1),
            strides=(2, 1),
            padding='same',
            use_bias=True,
        ),
    ])


@versioned_unhashable_object_fixture
def convtranspose_test2_int8_nnr301_inp_32x1x48_k3x1_oc48_s2x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(32, 1, 48)),
        tf.keras.layers.Conv2DTranspose(
            filters=48,
            kernel_size=(3, 1),
            strides=(2, 1),
            padding='same',
            use_bias=True,
        ),
    ])


@versioned_unhashable_object_fixture
def convtranspose_test3_int8_nnr301_inp_64x1x32_k3x1_oc32_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(64, 1, 32)),
        tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def convtranspose_test4_int8_nnr301_inp_64x1x32_k3x1_oc32_s2x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(64, 1, 32)),
        tf.keras.layers.Conv2DTranspose(
            filters=32,
            kernel_size=(3, 1),
            strides=(2, 1),
            padding='same',
            use_bias=True,
        ),
    ])


@versioned_unhashable_object_fixture
def convtranspose_test5_int8_nnr301_inp_128x1x32_k3x1_oc32_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(128, 1, 32)),
        tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def convtranspose_test6_int8_nnr301_inp_128x1x1_k3x1_oc1_s2x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(128, 1, 1)),
        tf.keras.layers.Conv2DTranspose(
            filters=1,
            kernel_size=(3, 1),
            strides=(2, 1),
            padding='same',
            use_bias=True,
        ),
    ])


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
