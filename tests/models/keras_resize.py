import tensorflow as tf

from torq.testing.versioned_fixtures import versioned_hashable_object_fixture, versioned_unhashable_object_fixture


def _resize_nearest_neighbor(inputs, size):
    # tf.compat.v1.image.resize_nearest_neighbor maps 1:1 to tfl.resize_nearest_neighbor,
    # unlike UpSampling2D which lowers to reshape+tile and loses the op under quantization.
    # Keras 3 functional models can't call a raw TF op on a KerasTensor, so wrap it in a Lambda.
    return tf.keras.layers.Lambda(
        lambda x: tf.compat.v1.image.resize_nearest_neighbor(
            x, size=size, align_corners=False, half_pixel_centers=False
        ),
        output_shape=(size[0], size[1], inputs.shape[-1]),
    )(inputs)

@versioned_unhashable_object_fixture
def model_resize_2x2():
    inputs = tf.keras.Input(shape=(8, 8, 32), batch_size=1)
    outputs = _resize_nearest_neighbor(inputs, size=(16, 16))
    return tf.keras.Model(inputs, outputs)

@versioned_unhashable_object_fixture
def model_resize_2x3():
    inputs = tf.keras.Input(shape=(5, 7, 16), batch_size=1)
    outputs = _resize_nearest_neighbor(inputs, size=(10, 21))
    return tf.keras.Model(inputs, outputs)


@versioned_unhashable_object_fixture
def model_resize_3x2():
    inputs = tf.keras.Input(shape=(6, 10, 24), batch_size=1)
    outputs = _resize_nearest_neighbor(inputs, size=(18, 20))
    return tf.keras.Model(inputs, outputs)


@versioned_unhashable_object_fixture
def model_resize_3x3():
    inputs = tf.keras.Input(shape=(7, 9, 8), batch_size=1)
    outputs = _resize_nearest_neighbor(inputs, size=(21, 27))
    return tf.keras.Model(inputs, outputs)


@versioned_unhashable_object_fixture
def model_resize_4x4():
    inputs = tf.keras.Input(shape=(4, 6, 64), batch_size=1)
    outputs = _resize_nearest_neighbor(inputs, size=(16, 24))
    return tf.keras.Model(inputs, outputs)


@versioned_unhashable_object_fixture
def model_resize_3x4():
    inputs = tf.keras.Input(shape=(9, 5, 32), batch_size=1)
    outputs = _resize_nearest_neighbor(inputs, size=(27, 20))
    return tf.keras.Model(inputs, outputs)


@versioned_unhashable_object_fixture
def model_resize_5x5():
    inputs = tf.keras.Input(shape=(3, 4, 12), batch_size=1)
    outputs = _resize_nearest_neighbor(inputs, size=(15, 20))
    return tf.keras.Model(inputs, outputs)