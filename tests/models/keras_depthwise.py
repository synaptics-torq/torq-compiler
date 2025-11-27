import tensorflow as tf

from torq.testing.versioned_fixtures import versioned_hashable_object_fixture, versioned_unhashable_object_fixture


@versioned_unhashable_object_fixture
def model010_depthwise_1x3x3x16_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(6, 42, 16)),
        tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            depth_multiplier=1,
            use_bias=True
        ),
        tf.keras.layers.ReLU()
    ])



@versioned_unhashable_object_fixture
def model011_depthwise_1x3x3x16_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(6, 4, 16)),
        tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            depth_multiplier=1,
            use_bias=True
        ),
        tf.keras.layers.ReLU()
    ])



@versioned_unhashable_object_fixture
def model012_depthwise_1x3x3x2_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(6, 6, 2)),
        tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            depth_multiplier=1,
            use_bias=True
        ),
        tf.keras.layers.ReLU()
    ])



@versioned_unhashable_object_fixture
def model013_depthwise_1x3x3x2_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(6, 6, 2)),
        tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            depth_multiplier=1,
            dilation_rate=(1, 1),
            use_bias=True
        ),
        tf.keras.layers.ReLU()
    ])



@versioned_unhashable_object_fixture
def model014_depthwise_1x3x3x8_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(13, 15, 8)),
        tf.keras.layers.DepthwiseConv2D(
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='valid',
            depth_multiplier=1,
            use_bias=True
        ),
        tf.keras.layers.ReLU()
    ])



