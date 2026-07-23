import tensorflow as tf

from torq.testing.versioned_fixtures import versioned_unhashable_object_fixture
from torq.testing.cases import Case


@versioned_unhashable_object_fixture
def conv2d_test1_int8_inp_256x1x1_k3x1_oc32_s2x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(256, 1, 1)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 1), strides=(2, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test2_int8_inp_128x1x32_k1x1_oc32_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(128, 1, 32)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test3_int8_inp_64x1x32_k1x1_oc64_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(64, 1, 32)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test4_int8_inp_64x1x64_k1x1_oc64_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(64, 1, 64)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test5_int8_inp_32x1x64_k1x1_oc64_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(32, 1, 64)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test6_int8_inp_16x1x64_k1x1_oc64_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(16, 1, 64)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test7_int8_inp_16x1x64_k1x1_oc32_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(16, 1, 64)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test8_int8_inp_16x1x32_k1x1_oc48_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(16, 1, 32)),
        tf.keras.layers.Conv2D(filters=48, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test9_int8_inp_16x1x48_k2x1_oc48_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(16, 1, 48)),
        tf.keras.layers.Conv2D(filters=48, kernel_size=(2, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test10_int8_inp_16x1x48_k1x1_oc48_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(16, 1, 48)),
        tf.keras.layers.Conv2D(filters=48, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test11_int8_inp_32x1x48_k1x1_oc48_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(32, 1, 48)),
        tf.keras.layers.Conv2D(filters=48, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test12_int8_inp_32x1x48_k2x1_oc48_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(32, 1, 48)),
        tf.keras.layers.Conv2D(filters=48, kernel_size=(2, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test13_int8_inp_64x1x48_k1x1_oc32_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(64, 1, 48)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test14_int8_inp_64x1x32_k3x1_oc32_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(64, 1, 32)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test15_int8_inp_64x1x32_k1x1_oc32_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(64, 1, 32)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test16_int8_inp_64x1x32_k2x1_oc32_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(64, 1, 32)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(2, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test17_int8_inp_128x1x32_k3x1_oc32_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(128, 1, 32)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test18_int8_inp_128x1x32_k1x1_oc1_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(128, 1, 32)),
        tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test19_int8_inp_128x1x1_k2x1_oc1_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(128, 1, 1)),
        tf.keras.layers.Conv2D(filters=1, kernel_size=(2, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test20_int8_inp_128x1x1_k1x1_oc1_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(128, 1, 1)),
        tf.keras.layers.Conv2D(filters=1, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


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

    return test_cases