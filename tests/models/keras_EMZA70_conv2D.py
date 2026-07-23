import tensorflow as tf

from torq.testing.cases import Case
from torq.testing.versioned_fixtures import versioned_unhashable_object_fixture


@versioned_unhashable_object_fixture
def conv2d_test1_int8_inp_136x160x1_k3x3_oc5_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(136, 160, 1)),
        tf.keras.layers.Conv2D(filters=5, kernel_size=(3, 3), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test2_int8_inp_67x79x5_k3x3_oc10_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(67, 79, 5)),
        tf.keras.layers.Conv2D(filters=10, kernel_size=(3, 3), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test3_int8_inp_30x36x10_k1x1_oc15_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(30, 36, 10)),
        tf.keras.layers.Conv2D(filters=15, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test4_int8_inp_15x18x15_k4x3_oc64_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(15, 18, 15)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(4, 3), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test5_int8_inp_12x16x64_k1x1_oc32_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(12, 16, 64)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test6_int8_inp_12x16x32_k1x1_oc4_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(12, 16, 32)),
        tf.keras.layers.Conv2D(filters=4, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test7_int8_inp_12x16x32_k1x1_oc1_s1x1_valid():
    tf.keras.utils.set_random_seed(43)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(12, 16, 32)),
        tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


def get_emza70_conv2d_test_cases():
    test_cases = []

    test_cases.append(Case("conv2d_test1_int8_inp_136x160x1_k3x3_oc5_s1x1_valid", {
        "keras_model_name": "conv2d_test1_int8_inp_136x160x1_k3x3_oc5_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test2_int8_inp_67x79x5_k3x3_oc10_s1x1_valid", {
        "keras_model_name": "conv2d_test2_int8_inp_67x79x5_k3x3_oc10_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test3_int8_inp_30x36x10_k1x1_oc15_s1x1_valid", {
        "keras_model_name": "conv2d_test3_int8_inp_30x36x10_k1x1_oc15_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test4_int8_inp_15x18x15_k4x3_oc64_s1x1_valid", {
        "keras_model_name": "conv2d_test4_int8_inp_15x18x15_k4x3_oc64_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test5_int8_inp_12x16x64_k1x1_oc32_s1x1_valid", {
        "keras_model_name": "conv2d_test5_int8_inp_12x16x64_k1x1_oc32_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test6_int8_inp_12x16x32_k1x1_oc4_s1x1_valid", {
        "keras_model_name": "conv2d_test6_int8_inp_12x16x32_k1x1_oc4_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test7_int8_inp_12x16x32_k1x1_oc1_s1x1_valid", {
        "keras_model_name": "conv2d_test7_int8_inp_12x16x32_k1x1_oc1_s1x1_valid"
    }))

    return test_cases
