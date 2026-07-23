import tensorflow as tf

from torq.testing.versioned_fixtures import versioned_unhashable_object_fixture
from torq.testing.cases import Case


@versioned_unhashable_object_fixture
def conv2d_test1_int8_mobilenetv2_inp_224x224x3_k3x3_oc32_s2x2_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test2_int8_mobilenetv2_inp_112x112x32_k1x1_oc16_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(112, 112, 32)),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test3_int8_mobilenetv2_inp_112x112x16_k1x1_oc96_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(112, 112, 16)),
        tf.keras.layers.Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test4_int8_mobilenetv2_inp_56x56x96_k1x1_oc24_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(56, 56, 96)),
        tf.keras.layers.Conv2D(filters=24, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test5_int8_mobilenetv2_inp_56x56x24_k1x1_oc144_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(56, 56, 24)),
        tf.keras.layers.Conv2D(filters=144, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test6_int8_mobilenetv2_inp_56x56x144_k1x1_oc24_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(56, 56, 144)),
        tf.keras.layers.Conv2D(filters=24, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test7_int8_mobilenetv2_inp_56x56x24_k1x1_oc144_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(56, 56, 24)),
        tf.keras.layers.Conv2D(filters=144, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test8_int8_mobilenetv2_inp_28x28x144_k1x1_oc32_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(28, 28, 144)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test9_int8_mobilenetv2_inp_28x28x32_k1x1_oc192_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(28, 28, 32)),
        tf.keras.layers.Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test10_int8_mobilenetv2_inp_28x28x192_k1x1_oc32_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(28, 28, 192)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test11_int8_mobilenetv2_inp_28x28x32_k1x1_oc192_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(28, 28, 32)),
        tf.keras.layers.Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test12_int8_mobilenetv2_inp_28x28x192_k1x1_oc32_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(28, 28, 192)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test13_int8_mobilenetv2_inp_28x28x32_k1x1_oc192_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(28, 28, 32)),
        tf.keras.layers.Conv2D(filters=192, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test14_int8_mobilenetv2_inp_14x14x192_k1x1_oc64_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(14, 14, 192)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test15_int8_mobilenetv2_inp_14x14x64_k1x1_oc384_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(14, 14, 64)),
        tf.keras.layers.Conv2D(filters=384, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test16_int8_mobilenetv2_inp_14x14x384_k1x1_oc64_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(14, 14, 384)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test17_int8_mobilenetv2_inp_14x14x64_k1x1_oc384_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(14, 14, 64)),
        tf.keras.layers.Conv2D(filters=384, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test18_int8_mobilenetv2_inp_14x14x384_k1x1_oc64_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(14, 14, 384)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test19_int8_mobilenetv2_inp_14x14x64_k1x1_oc384_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(14, 14, 64)),
        tf.keras.layers.Conv2D(filters=384, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test20_int8_mobilenetv2_inp_14x14x384_k1x1_oc64_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(14, 14, 384)),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test21_int8_mobilenetv2_inp_14x14x64_k1x1_oc384_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(14, 14, 64)),
        tf.keras.layers.Conv2D(filters=384, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test22_int8_mobilenetv2_inp_14x14x384_k1x1_oc96_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(14, 14, 384)),
        tf.keras.layers.Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test23_int8_mobilenetv2_inp_14x14x96_k1x1_oc576_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(14, 14, 96)),
        tf.keras.layers.Conv2D(filters=576, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test24_int8_mobilenetv2_inp_14x14x576_k1x1_oc96_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(14, 14, 576)),
        tf.keras.layers.Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test25_int8_mobilenetv2_inp_14x14x96_k1x1_oc576_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(14, 14, 96)),
        tf.keras.layers.Conv2D(filters=576, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test26_int8_mobilenetv2_inp_14x14x576_k1x1_oc96_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(14, 14, 576)),
        tf.keras.layers.Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test27_int8_mobilenetv2_inp_14x14x96_k1x1_oc576_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(14, 14, 96)),
        tf.keras.layers.Conv2D(filters=576, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test28_int8_mobilenetv2_inp_7x7x576_k1x1_oc160_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(7, 7, 576)),
        tf.keras.layers.Conv2D(filters=160, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test29_int8_mobilenetv2_inp_7x7x160_k1x1_oc960_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(7, 7, 160)),
        tf.keras.layers.Conv2D(filters=960, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test30_int8_mobilenetv2_inp_7x7x960_k1x1_oc160_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(7, 7, 960)),
        tf.keras.layers.Conv2D(filters=160, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test31_int8_mobilenetv2_inp_7x7x160_k1x1_oc960_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(7, 7, 160)),
        tf.keras.layers.Conv2D(filters=960, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test32_int8_mobilenetv2_inp_7x7x960_k1x1_oc160_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(7, 7, 960)),
        tf.keras.layers.Conv2D(filters=160, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test33_int8_mobilenetv2_inp_7x7x160_k1x1_oc960_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(7, 7, 160)),
        tf.keras.layers.Conv2D(filters=960, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test34_int8_mobilenetv2_inp_7x7x960_k1x1_oc320_s1x1_same():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(7, 7, 960)),
        tf.keras.layers.Conv2D(filters=320, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True),
    ])


@versioned_unhashable_object_fixture
def conv2d_test35_int8_mobilenetv2_inp_7x7x320_k1x1_oc1280_s1x1_valid():
    tf.keras.utils.set_random_seed(42)
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(7, 7, 320)),
        tf.keras.layers.Conv2D(filters=1280, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True),
    ])


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

    return test_cases
