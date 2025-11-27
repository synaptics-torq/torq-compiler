import tensorflow as tf

from torq.testing.versioned_fixtures import versioned_hashable_object_fixture, versioned_unhashable_object_fixture


@versioned_unhashable_object_fixture
def model089_softmax_inp1x1916x2():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1916, 2)),
        tf.keras.layers.Softmax()
    ])



