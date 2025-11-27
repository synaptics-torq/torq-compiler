import tensorflow as tf

from torq.testing.versioned_fixtures import versioned_hashable_object_fixture, versioned_unhashable_object_fixture


@versioned_unhashable_object_fixture
def model110_transpose_last2first():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=10, shape=(8, 32, 48), dtype=tf.float32),
        tf.keras.layers.Permute((3, 1, 2))
    ])



@versioned_unhashable_object_fixture
def model111_transpose_first2last():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=10, shape=(8, 32, 48), name="serving_default_args_0_0"),
        tf.keras.layers.Permute((2, 3, 1))
    ])



@versioned_unhashable_object_fixture
def model112_transpose_2d():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    # The model performs a transpose operation, which can be implemented with a Permute layer.
    # The input shape is [10, 8, 32, 48] and the output is [10, 32, 8, 48].
    # This corresponds to a permutation of axes (0, 2, 1, 3).
    # The Permute layer's `dims` argument applies to the non-batch dimensions.
    # So, for an input of shape (batch, 8, 32, 48), the permutation (2, 1, 3)
    # on the non-batch axes (1, 2, 3) results in an output shape of (batch, 32, 8, 48).
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=10, shape=(8, 32, 48)),
        tf.keras.layers.Permute((2, 1, 3))
    ])



@versioned_unhashable_object_fixture
def model160_transpose_ChLastToFirst_1x8x10x4():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(8, 10, 4), dtype=tf.float32),
        tf.keras.layers.Permute((3, 1, 2))
    ])



@versioned_unhashable_object_fixture
def model161_transpose_ChFirstToLast_1x8x10x4():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(8, 10, 4), dtype=tf.float32),
        tf.keras.layers.Permute((2, 3, 1))
    ])



@versioned_unhashable_object_fixture
def model162_transpose_2d_matrix_1x4x6x2():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(4, 6, 2)),
        tf.keras.layers.Permute((2, 1, 3))
    ])



