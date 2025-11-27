import tensorflow as tf

from torq.testing.versioned_fixtures import versioned_hashable_object_fixture, versioned_unhashable_object_fixture


@versioned_unhashable_object_fixture
def model001_conv_3X3_small():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(24, 32, 1)),
        tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), strides=(2, 2), padding="same", use_bias=True),
        tf.keras.layers.ReLU(6.0)
    ])



@versioned_unhashable_object_fixture
def model002_conv_2x3_small_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(3, 4, 2)),
        tf.keras.layers.Conv2D(filters=3, kernel_size=(2, 3), strides=(1, 1), padding='valid', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model004_conv_3x3_valid_bias():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(3, 4, 2)),
        tf.keras.layers.Conv2D(filters=3, kernel_size=(3, 3), strides=(1, 1), padding='valid', use_bias=True),
        tf.keras.layers.ReLU()
    ])



@versioned_unhashable_object_fixture
def model005_conv2d_4x3x3x3_padding():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(3, 4, 3)),
        tf.keras.layers.Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=True),
        tf.keras.layers.ReLU()
    ])



@versioned_unhashable_object_fixture
def model006_conv2d_4x6x6x1_pad_stride2():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(13, 35, 1), name="input_1"),
        tf.keras.layers.Conv2D(filters=4, kernel_size=(6, 6), strides=(2, 2), padding='same', use_bias=True),
        tf.keras.layers.ReLU()
    ])



@versioned_unhashable_object_fixture
def model007_conv2d_8x3x3x4():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(12, 14, 4)),
        tf.keras.layers.Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True),
        tf.keras.layers.ReLU()
    ])



@versioned_unhashable_object_fixture
def model008_conv2d_4x4x4x4():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(12, 14, 4)),
        tf.keras.layers.Conv2D(filters=4, kernel_size=(4, 4), strides=(1, 1), padding='same', use_bias=True),
        tf.keras.layers.ReLU()
    ])



@versioned_unhashable_object_fixture
def model009_conv_3x3_same_bias():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 4, 1)),
        tf.keras.layers.Conv2D(filters=1, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True),
        tf.keras.layers.ReLU()
    ])



@versioned_unhashable_object_fixture
def model200_conv_4_4_5_19_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(7, 30, 19)),
        tf.keras.layers.Conv2D(filters=4, kernel_size=(4, 5), strides=(1, 1), padding='valid', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model201_conv_20x4x1x12_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(38, 50, 12)),
        tf.keras.layers.Conv2D(
            filters=20,
            kernel_size=(4, 1),
            strides=(1, 1),
            padding='valid',
            use_bias=True
        )
    ])



@versioned_unhashable_object_fixture
def model202_conv_20x1x2x14_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(39, 26, 14)),
        tf.keras.layers.Conv2D(filters=20, kernel_size=(1, 2), strides=(1, 1), padding='same', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model203_conv_19x5x2x24_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(58, 9, 24)),
        tf.keras.layers.Conv2D(filters=19, kernel_size=(5, 2), strides=(1, 1), padding='same', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model204_conv_32x5x4x5_valid_stride4():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(41, 19, 5), name="conv2d_input_int8"),
        tf.keras.layers.Conv2D(filters=32, kernel_size=(5, 4), strides=(4, 4), padding='valid', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model205_conv_9x2x5x21_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(22, 55, 21)),
        tf.keras.layers.Conv2D(filters=9, kernel_size=(2, 5), strides=(1, 1), padding='same', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model206_conv_17x1x1x19_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(18, 26, 19)),
        tf.keras.layers.Conv2D(filters=17, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model207_conv_25x2x1x26_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 3, 26)),
        tf.keras.layers.Conv2D(filters=25, kernel_size=(2, 1), strides=(1, 1), padding='same', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model208_conv_13x5x3x7_same_stride3():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(16, 55, 7)),
        tf.keras.layers.Conv2D(filters=13, kernel_size=(5, 3), strides=(3, 3), padding="same", use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model209_conv_25x1x5x9_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(20, 42, 9)),
        tf.keras.layers.Conv2D(filters=25, kernel_size=(1, 5), strides=(1, 1), padding='same', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model210_conv_12x3x3x24_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(5, 40, 24)),
        tf.keras.layers.Conv2D(filters=12, kernel_size=(4, 3), strides=(1, 1), padding='same', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model211_conv_26x5x4x2_valid_stride3():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(34, 6, 2)),
        tf.keras.layers.Conv2D(filters=26, kernel_size=(5, 4), strides=(3, 3), padding='valid', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model212_conv_15x4x2x29_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(13, 16, 29)),
        tf.keras.layers.Conv2D(filters=15, kernel_size=(4, 2), strides=(1, 1), padding='valid', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model213_conv_22x1x3x9_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(60, 12, 9)),
        tf.keras.layers.Conv2D(filters=22, kernel_size=(1, 3), strides=(1, 1), padding='valid', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model214_conv_8x4x4x27_same_stride3():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(45, 22, 27)),
        tf.keras.layers.Conv2D(filters=8, kernel_size=(4, 4), strides=(3, 3), padding='same', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model215_conv_16x5x2x7_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(54, 63, 7)),
        tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 2), strides=(1, 1), padding='valid', activation=None, use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model216_conv_9x4x3x11_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(6, 46, 11)),
        tf.keras.layers.Conv2D(filters=9, kernel_size=(4, 3), strides=(1, 1), padding='same', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model217_conv_17x3x5x10_valid_stride3():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(15, 13, 10)),
        tf.keras.layers.Conv2D(filters=17, kernel_size=(3, 5), strides=(3, 3), padding='valid', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model218_conv_23x5x4x10_valid_stride2():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(12, 52, 10)),
        tf.keras.layers.Conv2D(filters=23, kernel_size=(5, 4), strides=(2, 2), padding='valid', use_bias=True, activation=None)
    ])



@versioned_unhashable_object_fixture
def model219_conv_26x3x3x20_same_stride3():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(43, 61, 20)),
        tf.keras.layers.Conv2D(filters=26, kernel_size=(3, 3), strides=(3, 3), padding='same', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model220_conv_inp1x4x4x2_ker3x3_same_ker2x2_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 4, 2)),
        tf.keras.layers.Conv2D(filters=2, kernel_size=(3, 3), strides=(1, 1), padding="same", use_bias=True),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters=2, kernel_size=(2, 2), strides=(1, 1), padding="same", use_bias=True),
        tf.keras.layers.ReLU()
    ])



@versioned_unhashable_object_fixture
def model221_conv_inp1x4x4x2_ker3x3_same_ker3x3_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 4, 2)),
        tf.keras.layers.Conv2D(filters=2, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters=2, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True),
        tf.keras.layers.ReLU()
    ])



@versioned_unhashable_object_fixture
def model222_conv_inp1x4x4x2_ker2x2_same_ker3x3_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 4, 2)),
        tf.keras.layers.Conv2D(filters=2, kernel_size=(2, 2), strides=(1, 1), padding='same', use_bias=True),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Conv2D(filters=2, kernel_size=(3, 3), strides=(1, 1), padding='same', use_bias=True),
        tf.keras.layers.ReLU()
    ])



@versioned_unhashable_object_fixture
def model237_conv_inp1x18x76x24_1x5x2_same_stride1x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(18, 76, 24)),
        tf.keras.layers.Conv2D(filters=1, kernel_size=(5, 2), strides=(1, 1), padding='same', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model238_conv_inp1x58x26x24_1x5x2_same_stride1x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(58, 26, 24)),
        tf.keras.layers.Conv2D(filters=1, kernel_size=(5, 2), strides=(1, 1), padding='same', use_bias=True)
    ])



