import tensorflow as tf

from dataclasses import dataclass
from torq.testing.versioned_fixtures import versioned_hashable_object_fixture, versioned_unhashable_object_fixture


@dataclass
class ConvModelParams:
    width: int
    height: int
    filters: int
    kernel_size: int
    input_channels: int

    def idfn(val):
        return f"w{val.width}_h{val.height}_f{val.filters}_k{val.kernel_size}_c{val.input_channels}"


conv_model_params = [ConvModelParams(12, 12, 5, 1, 64), ConvModelParams(100, 100, 5, 6, 4)]


@versioned_hashable_object_fixture
def keras_model_params(case_config):
    return case_config['keras_model_params']


@versioned_unhashable_object_fixture
def conv_model(keras_model_params):

    # make sure tests use reproducible weights
    tf.keras.utils.set_random_seed(21321)

    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(keras_model_params.height, keras_model_params.width, keras_model_params.input_channels)),
        tf.keras.layers.Conv2D(filters=keras_model_params.filters, kernel_size=keras_model_params.kernel_size, padding='same')
    ])


@versioned_unhashable_object_fixture
def conv_act_model():

    # make sure tests use reproducible weights
    tf.keras.utils.set_random_seed(21321)

    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(64, 64, 3)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same'),
        tf.keras.layers.ReLU(6.0)
    ])


@versioned_unhashable_object_fixture
def transpose_conv_model():

    # make sure tests use reproducible weights
    tf.keras.utils.set_random_seed(21321)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(8, 8, 4)),
        tf.keras.layers.Conv2DTranspose(filters=4, kernel_size=4, strides=2, padding='same')
    ])
