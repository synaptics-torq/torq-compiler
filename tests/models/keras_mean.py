import tensorflow as tf

from torq.testing.versioned_fixtures import versioned_hashable_object_fixture, versioned_unhashable_object_fixture


@versioned_unhashable_object_fixture
def model180_mean_12x8x256_to_1x256():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(12, 8, 256)),
        tf.keras.layers.GlobalAveragePooling2D()
    ])



@versioned_unhashable_object_fixture
def model181_mean_inp_4x8x6_zp_64():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 8, 6), name='x'),
        tf.keras.layers.GlobalAveragePooling2D()
    ])



@versioned_unhashable_object_fixture
def model182_mean_inp_22x32x16_zp_1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(23, 32, 16)),
        tf.keras.layers.GlobalAveragePooling2D()
    ])



@versioned_unhashable_object_fixture
def model183_mean_inp_37x17x11_zp_minus127():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(37, 17, 11)),
        tf.keras.layers.GlobalAveragePooling2D()
    ])



@versioned_unhashable_object_fixture
def model184_mean_inp_12x16_64_zp_minus76():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(12, 16, 64)),
        tf.keras.layers.GlobalAveragePooling2D()
    ])



@versioned_unhashable_object_fixture
def model185_mean_inp_32x1x512_zp_74_scale_3():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(32, 1, 512)),
        tf.keras.layers.GlobalAveragePooling2D()
    ])



@versioned_unhashable_object_fixture
def model186_mean_53x37x1_zp_minus25_scale_39():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(53, 37, 1)),
        tf.keras.layers.GlobalAveragePooling2D()
    ])



@versioned_unhashable_object_fixture
def model187_mean_1x32x512_zp_26():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1, 32, 512)),
        tf.keras.layers.GlobalAveragePooling2D()
    ])



@versioned_unhashable_object_fixture
def model188_mean_11x1x1_zp_52_scale_em7():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(11, 1, 1), name="x"),
        tf.keras.layers.GlobalAveragePooling2D()
    ])



@versioned_unhashable_object_fixture
def model189_mean_1x173x1_zp_minus71_scale19():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    inputs = tf.keras.layers.Input(batch_size=1, shape=(1, 173, 1), name='x')
    x = tf.keras.layers.GlobalAveragePooling2D()(inputs)
    return tf.keras.Model(inputs=inputs, outputs=x)



