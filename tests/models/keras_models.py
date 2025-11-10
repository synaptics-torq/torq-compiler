import tensorflow as tf

from dataclasses import dataclass

@dataclass
class ConvModelParams:
    width: int
    height: int
    filters: int
    kernel_size: int
    input_channels: int

    @staticmethod
    def idfn(val):
        return f"w{val.width}_h{val.height}_f{val.filters}_k{val.kernel_size}_c{val.input_channels}"


def conv_model(params: ConvModelParams):

    # make sure tests use reproducible weights
    tf.keras.utils.set_random_seed(21321)

    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(params.height, params.width, params.input_channels)),
        tf.keras.layers.Conv2D(filters=params.filters, kernel_size=params.kernel_size, padding='same')
    ])


def conv_act_model():

    # make sure tests use reproducible weights
    tf.keras.utils.set_random_seed(21321)

    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(64, 64, 3)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same'),
        tf.keras.layers.ReLU(6.0)
    ])


def transpose_conv_model():

    # make sure tests use reproducible weights
    tf.keras.utils.set_random_seed(21321)
    
    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(8, 8, 4)),
        tf.keras.layers.Conv2DTranspose(filters=4, kernel_size=4, strides=2, padding='same')
    ])