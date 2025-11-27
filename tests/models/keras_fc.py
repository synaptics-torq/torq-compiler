import tensorflow as tf

from torq.testing.versioned_fixtures import versioned_hashable_object_fixture, versioned_unhashable_object_fixture


@versioned_unhashable_object_fixture
def model003_hello_world():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1,)),
        tf.keras.layers.Dense(units=16),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(units=16),
        tf.keras.layers.ReLU(),
        tf.keras.layers.Dense(units=1)
    ])



@versioned_unhashable_object_fixture
def model020_fc_1x1():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1,)),
        tf.keras.layers.Dense(units=1),
        tf.keras.layers.ReLU(6.0)
    ])



@versioned_unhashable_object_fixture
def model021_fc_1991x61():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(61,)),
        tf.keras.layers.Dense(units=1991),
        tf.keras.layers.ReLU()
    ])



@versioned_unhashable_object_fixture
def model022_fc_1024x1024():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1024,)),
        tf.keras.layers.Dense(units=1024)
    ])



@versioned_unhashable_object_fixture
def model023_fc_512x1000():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1000,)),
        tf.keras.layers.Dense(units=512),
        tf.keras.layers.ReLU(6.0)
    ])



@versioned_unhashable_object_fixture
def model024_fc_97x2000():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(2000,)),
        tf.keras.layers.Dense(units=97),
        tf.keras.layers.ReLU(6.0)
    ])



@versioned_unhashable_object_fixture
def model026_fc_1991x61():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(61,)),
        tf.keras.layers.Dense(units=1991),
        tf.keras.layers.ReLU()
    ])



@versioned_unhashable_object_fixture
def model027_fc_500x700():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(700,)),
        tf.keras.layers.Dense(units=500),
        tf.keras.layers.Activation('sigmoid')
    ])



@versioned_unhashable_object_fixture
def model029_fc_1000x1000():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(1000,)),
        tf.keras.layers.Dense(units=1000),
        tf.keras.layers.Softmax()
    ])



@versioned_unhashable_object_fixture
def model080_fc_7x3():
    """
    Reconstructed model based on TFLite analysis.
    Uses reproducible weights for deterministic quantization.
    """
    # Ensure reproducible weights
    tf.keras.utils.set_random_seed(42)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(3,)),
        tf.keras.layers.Dense(units=7),
        tf.keras.layers.ReLU()
    ])



