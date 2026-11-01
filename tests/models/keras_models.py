import tensorflow as tf

from torq.testing.tensorflow import keras_model_fixture


@keras_model_fixture
def conv_act_model(request):

    # make sure tests use reproducible weights
    tf.keras.utils.set_random_seed(21321)

    return tf.keras.Sequential([
        tf.keras.layers.Input(batch_size=1, shape=(64, 64, 3)),
        tf.keras.layers.Conv2D(filters=32, kernel_size=3, padding='same'),
        tf.keras.layers.ReLU(6.0)
    ])
