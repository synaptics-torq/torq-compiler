import tensorflow as tf

from torq.testing.cases import Case
from torq.testing.versioned_fixtures import versioned_unhashable_object_fixture


@versioned_unhashable_object_fixture
def conv2d_test1_int8_inp_242x137x3_k3x3_oc16_s2x2_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(242, 137, 3)),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test2_int8_inp_120x68x16_k1x1_oc32_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(120, 68, 16)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test3_int8_inp_60x34x32_k1x1_oc32_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(60, 34, 32)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test4_int8_inp_60x34x32_k1x1_oc32_s1x1_valid():
    tf.keras.utils.set_random_seed(43)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(60, 34, 32)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test5_int8_inp_30x17x32_k1x1_oc64_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(30, 17, 32)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test6_int8_inp_30x17x64_k1x1_oc64_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(30, 17, 64)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test7_int8_inp_30x17x64_k1x1_oc64_s1x1_valid():
    tf.keras.utils.set_random_seed(43)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(30, 17, 64)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test8_int8_inp_30x17x64_k1x1_oc64_s1x1_valid():
    tf.keras.utils.set_random_seed(44)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(30, 17, 64)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test9_int8_inp_30x17x64_k1x1_oc64_s1x1_valid():
    tf.keras.utils.set_random_seed(45)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(30, 17, 64)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test10_int8_inp_15x9x64_k1x1_oc128_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(15, 9, 64)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test11_int8_inp_15x9x128_k1x1_oc128_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(15, 9, 128)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test12_int8_inp_15x9x128_k1x1_oc128_s1x1_valid():
    tf.keras.utils.set_random_seed(43)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(15, 9, 128)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test13_int8_inp_8x5x128_k1x1_oc256_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(8, 5, 128)),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test14_int8_inp_8x5x256_k1x1_oc256_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(8, 5, 256)),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test15_int8_inp_8x5x256_k1x1_oc4_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(8, 5, 256)),
        tf.keras.layers.Conv2D(filters=4, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test16_int8_inp_8x5x256_k1x1_oc64_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(8, 5, 256)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test17_int8_inp_4x3x64_k1x1_oc256_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 3, 64)),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test18_int8_inp_4x3x256_k3x3_oc6_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 3, 256)),
        tf.keras.layers.Conv2D(filters=6, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test19_int8_inp_4x3x256_k3x3_oc12_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 3, 256)),
        tf.keras.layers.Conv2D(filters=12, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test20_int8_inp_4x3x256_k3x3_oc9_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 3, 256)),
        tf.keras.layers.Conv2D(filters=9, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test21_int8_inp_8x5x256_k1x1_oc8_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(8, 5, 256)),
        tf.keras.layers.Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test22_int8_inp_8x5x256_k1x1_oc6_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(8, 5, 256)),
        tf.keras.layers.Conv2D(filters=6, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test23_int8_inp_15x9x128_k1x1_oc4_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(15, 9, 128)),
        tf.keras.layers.Conv2D(filters=4, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test24_int8_inp_15x9x128_k1x1_oc8_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(15, 9, 128)),
        tf.keras.layers.Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test25_int8_inp_15x9x128_k1x1_oc6_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(15, 9, 128)),
        tf.keras.layers.Conv2D(filters=6, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test26_int8_inp_30x17x64_k1x1_oc6_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(30, 17, 64)),
        tf.keras.layers.Conv2D(filters=6, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test27_int8_inp_30x17x64_k1x1_oc12_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(30, 17, 64)),
        tf.keras.layers.Conv2D(filters=12, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test28_int8_inp_30x17x64_k1x1_oc9_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(30, 17, 64)),
        tf.keras.layers.Conv2D(filters=9, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


def get_emza75_conv2d_test_cases():
    test_cases = []

    test_cases.append(Case("conv2d_test1_int8_inp_242x137x3_k3x3_oc16_s2x2_valid", {
        "keras_model_name": "conv2d_test1_int8_inp_242x137x3_k3x3_oc16_s2x2_valid"
    }))
    test_cases.append(Case("conv2d_test2_int8_inp_120x68x16_k1x1_oc32_s1x1_valid", {
        "keras_model_name": "conv2d_test2_int8_inp_120x68x16_k1x1_oc32_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test3_int8_inp_60x34x32_k1x1_oc32_s1x1_valid", {
        "keras_model_name": "conv2d_test3_int8_inp_60x34x32_k1x1_oc32_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test4_int8_inp_60x34x32_k1x1_oc32_s1x1_valid", {
        "keras_model_name": "conv2d_test4_int8_inp_60x34x32_k1x1_oc32_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test5_int8_inp_30x17x32_k1x1_oc64_s1x1_valid", {
        "keras_model_name": "conv2d_test5_int8_inp_30x17x32_k1x1_oc64_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test6_int8_inp_30x17x64_k1x1_oc64_s1x1_valid", {
        "keras_model_name": "conv2d_test6_int8_inp_30x17x64_k1x1_oc64_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test7_int8_inp_30x17x64_k1x1_oc64_s1x1_valid", {
        "keras_model_name": "conv2d_test7_int8_inp_30x17x64_k1x1_oc64_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test8_int8_inp_30x17x64_k1x1_oc64_s1x1_valid", {
        "keras_model_name": "conv2d_test8_int8_inp_30x17x64_k1x1_oc64_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test9_int8_inp_30x17x64_k1x1_oc64_s1x1_valid", {
        "keras_model_name": "conv2d_test9_int8_inp_30x17x64_k1x1_oc64_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test10_int8_inp_15x9x64_k1x1_oc128_s1x1_valid", {
        "keras_model_name": "conv2d_test10_int8_inp_15x9x64_k1x1_oc128_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test11_int8_inp_15x9x128_k1x1_oc128_s1x1_valid", {
        "keras_model_name": "conv2d_test11_int8_inp_15x9x128_k1x1_oc128_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test12_int8_inp_15x9x128_k1x1_oc128_s1x1_valid", {
        "keras_model_name": "conv2d_test12_int8_inp_15x9x128_k1x1_oc128_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test13_int8_inp_8x5x128_k1x1_oc256_s1x1_valid", {
        "keras_model_name": "conv2d_test13_int8_inp_8x5x128_k1x1_oc256_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test14_int8_inp_8x5x256_k1x1_oc256_s1x1_valid", {
        "keras_model_name": "conv2d_test14_int8_inp_8x5x256_k1x1_oc256_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test15_int8_inp_8x5x256_k1x1_oc4_s1x1_valid", {
        "keras_model_name": "conv2d_test15_int8_inp_8x5x256_k1x1_oc4_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test16_int8_inp_8x5x256_k1x1_oc64_s1x1_same", {
        "keras_model_name": "conv2d_test16_int8_inp_8x5x256_k1x1_oc64_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test17_int8_inp_4x3x64_k1x1_oc256_s1x1_valid", {
        "keras_model_name": "conv2d_test17_int8_inp_4x3x64_k1x1_oc256_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test18_int8_inp_4x3x256_k3x3_oc6_s1x1_same", {
        "keras_model_name": "conv2d_test18_int8_inp_4x3x256_k3x3_oc6_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test19_int8_inp_4x3x256_k3x3_oc12_s1x1_same", {
        "keras_model_name": "conv2d_test19_int8_inp_4x3x256_k3x3_oc12_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test20_int8_inp_4x3x256_k3x3_oc9_s1x1_same", {
        "keras_model_name": "conv2d_test20_int8_inp_4x3x256_k3x3_oc9_s1x1_same"
    }))
    test_cases.append(Case("conv2d_test21_int8_inp_8x5x256_k1x1_oc8_s1x1_valid", {
        "keras_model_name": "conv2d_test21_int8_inp_8x5x256_k1x1_oc8_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test22_int8_inp_8x5x256_k1x1_oc6_s1x1_valid", {
        "keras_model_name": "conv2d_test22_int8_inp_8x5x256_k1x1_oc6_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test23_int8_inp_15x9x128_k1x1_oc4_s1x1_valid", {
        "keras_model_name": "conv2d_test23_int8_inp_15x9x128_k1x1_oc4_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test24_int8_inp_15x9x128_k1x1_oc8_s1x1_valid", {
        "keras_model_name": "conv2d_test24_int8_inp_15x9x128_k1x1_oc8_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test25_int8_inp_15x9x128_k1x1_oc6_s1x1_valid", {
        "keras_model_name": "conv2d_test25_int8_inp_15x9x128_k1x1_oc6_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test26_int8_inp_30x17x64_k1x1_oc6_s1x1_valid", {
        "keras_model_name": "conv2d_test26_int8_inp_30x17x64_k1x1_oc6_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test27_int8_inp_30x17x64_k1x1_oc12_s1x1_valid", {
        "keras_model_name": "conv2d_test27_int8_inp_30x17x64_k1x1_oc12_s1x1_valid"
    }))
    test_cases.append(Case("conv2d_test28_int8_inp_30x17x64_k1x1_oc9_s1x1_valid", {
        "keras_model_name": "conv2d_test28_int8_inp_30x17x64_k1x1_oc9_s1x1_valid"
    }))

    return test_cases
