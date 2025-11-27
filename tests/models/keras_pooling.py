import tensorflow as tf

from torq.testing.versioned_fixtures import versioned_hashable_object_fixture, versioned_unhashable_object_fixture


@versioned_unhashable_object_fixture
def model030_avgpool_inp1x10x10x4_pool3x3_stride3x3_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(3, 3), padding='valid')
    ])



@versioned_unhashable_object_fixture
def model031_avgpool_inp1x10x10x4_pool4x4_stride4x4_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')
    ])



@versioned_unhashable_object_fixture
def model032_avgpool_inp1x10x10x4_pool5x5_stride5x5_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4), dtype=tf.float32),
        tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(5, 5), padding='valid')
    ])



@versioned_unhashable_object_fixture
def model033_avgpool_inp1x10x10x4_pool5x5_stride1x1_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(5, 5), padding='valid')
    ])



@versioned_unhashable_object_fixture
def model034_avgpool_inp1x10x10x4_pool1x3_stride1x3_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4), dtype=tf.float32),
        tf.keras.layers.AveragePooling2D(pool_size=(1, 3), strides=(1, 3), padding='valid')
    ])



@versioned_unhashable_object_fixture
def model035_avgpool_inp1x10x10x4_pool3x3_stride3x3_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.AveragePooling2D(pool_size=(3, 3), strides=(3, 3), padding='same')
    ])



@versioned_unhashable_object_fixture
def model036_avgpool_inp1x10x10x4_pool4x4_stride4x4_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.AveragePooling2D(pool_size=(4, 4), strides=(4, 4), padding='same')
    ])



@versioned_unhashable_object_fixture
def model037_avgpool_inp1x10x10x4_pool5x5_stride5x5_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(5, 5), padding='same')
    ])



@versioned_unhashable_object_fixture
def model038_avgpool_inp1x10x10x4_pool5x5_stride1x1_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.AveragePooling2D(pool_size=(5, 5), strides=(5, 5), padding='same')
    ])



@versioned_unhashable_object_fixture
def model039_avgpool_inp1x10x10x4_pool1x3_stride1x1_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.AveragePooling2D(pool_size=(1, 3), strides=(1, 3), padding="same")
    ])



@versioned_unhashable_object_fixture
def model040_maxpool_inp1x10x10x4_pool3x3_stride3x3_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='valid')
    ])



@versioned_unhashable_object_fixture
def model041_maxpool_inp1x10x10x4_pool4x4_stride4x4_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding='valid')
    ])



@versioned_unhashable_object_fixture
def model042_maxpool_inp1x10x10x4_pool5x5_stride5x5_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='valid')
    ])



@versioned_unhashable_object_fixture
def model043_maxpool_inp1x10x10x4_pool5x5_stride1x1_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='valid')
    ])



@versioned_unhashable_object_fixture
def model044_maxpool_inp1x10x10x4_pool1x3_stride1x3_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 3), padding='valid')
    ])



@versioned_unhashable_object_fixture
def model045_maxpool_inp1x10x10x4_pool3x3_stride3x3_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(3, 3), padding='same')
    ])



@versioned_unhashable_object_fixture
def model046_maxpool_inp1x10x10x4_pool4x4_stride4x4_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.MaxPooling2D(pool_size=(4, 4), strides=(4, 4), padding="same")
    ])



@versioned_unhashable_object_fixture
def model047_maxpool_inp1x10x10x4_pool5x5_stride5x5_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding='same')
    ])



@versioned_unhashable_object_fixture
def model048_maxpool_inp1x10x10x4_pool5x5_stride1x1_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4), dtype=tf.float32),
        tf.keras.layers.MaxPooling2D(pool_size=(5, 5), strides=(5, 5), padding="same")
    ])



@versioned_unhashable_object_fixture
def model049_maxpool_inp1x10x10x4_pool1x3_stride1x1_same():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.MaxPooling2D(pool_size=(1, 3), strides=(1, 3), padding='same')
    ])



@versioned_unhashable_object_fixture
def model055_avgpool_inp1x10x10x4_pool2x2_stride2x2_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(10, 10, 4)),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
    ])



@versioned_unhashable_object_fixture
def model056_avgpool_inp1x100x100x1_pool2x2_stride2x2_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(100, 100, 1)),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
    ])



@versioned_unhashable_object_fixture
def model057_avgpool_inp1x100x100x4_pool2x2_stride2x2_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(100, 100, 4)),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
    ])



@versioned_unhashable_object_fixture
def model058_avgpool_inp1x134x158x5_pool2x2_stride2x2_valid():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(134, 158, 5)),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
    ])



@versioned_unhashable_object_fixture
def model065_SmartAve_1x25x5x4():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(25, 5, 4)),
        tf.keras.layers.AveragePooling2D(pool_size=(25, 5), strides=(25, 5), padding='valid')
    ])



@versioned_unhashable_object_fixture
def model066_SmartAve_1x5x25x4():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(5, 25, 4)),
        tf.keras.layers.AveragePooling2D(pool_size=(5, 25), strides=(5, 25), padding='valid')
    ])



@versioned_unhashable_object_fixture
def model117_avgpool_valid_model70_L1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(134, 158, 5)),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
    ])



@versioned_unhashable_object_fixture
def model118_avgpool_valid_model70_L3():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(65, 77, 10)),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
    ])



@versioned_unhashable_object_fixture
def model119_avgpool_valid_model70_L6():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(30, 36, 15)),
        tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')
    ])



