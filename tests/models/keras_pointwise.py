import tensorflow as tf

from torq.testing.versioned_fixtures import versioned_hashable_object_fixture, versioned_unhashable_object_fixture


@versioned_unhashable_object_fixture
def model016_pointwise_8x8x8x16_stride2x2():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(8, 8, 16), name="serving_default_args_0_0"),
        tf.keras.layers.Conv2D(filters=8, kernel_size=(1, 1), strides=(2, 2), padding='same', use_bias=True, activation=None)
    ])



@versioned_unhashable_object_fixture
def model017_pointwise_8x8x9x16_stride2x2():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(8, 9, 16)),
        tf.keras.layers.Conv2D(filters=8, kernel_size=(1, 1), strides=(2, 2), padding='same', activation=None, use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model018_pointwise_8x9x8x16_stride2x2():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(9, 8, 16)),
        tf.keras.layers.Conv2D(filters=8, kernel_size=(1, 1), strides=(2, 2), padding="same", activation=None, use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model019_pointwise_8x9x9x16_stride2x2():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(9, 9, 16)),
        tf.keras.layers.Conv2D(filters=8, kernel_size=(1, 1), strides=(2, 2), padding='same', use_bias=True, activation=None)
    ])



@versioned_unhashable_object_fixture
def model113_pointwise_1x1x1024():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(8, 8, 1024)),
        tf.keras.layers.Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding='same', use_bias=True)
    ])



@versioned_unhashable_object_fixture
def model240_pointwise_inp1x4x4x16_8x1x1_valid_stride1x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 4, 16)),
        tf.keras.layers.Conv2D(filters=8, kernel_size=(1, 1), strides=(1, 1), padding='valid', use_bias=True, activation=None)
    ])



