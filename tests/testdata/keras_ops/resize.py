"""Keras test models: nearest-neighbor resize (TFLite RESIZE_NEAREST_NEIGHBOR)."""

import tensorflow as tf

from torq.testing.keras_builder import functional


def _resize(shape, size):
    # tf.compat.v1.image.resize_nearest_neighbor maps 1:1 to tfl.resize_nearest_neighbor,
    # unlike UpSampling2D which lowers to reshape+tile and loses the op under quantization.
    # Keras 3 functional models can't call a raw TF op on a KerasTensor, so wrap it in a Lambda.
    def build(x):
        return tf.keras.layers.Lambda(
            lambda t: tf.compat.v1.image.resize_nearest_neighbor(
                t, size=size, align_corners=False, half_pixel_centers=False
            ),
            output_shape=(size[0], size[1], shape[-1]),
        )(x)
    return functional(shape, build)


CASES = {
    "model_resize_2x3": _resize((5, 7, 16), (10, 21)),
    "model_resize_3x2": _resize((6, 10, 24), (18, 20)),
    "model_resize_3x3": _resize((7, 9, 8), (21, 27)),
    "model_resize_4x4": _resize((4, 6, 64), (16, 24)),
    "model_resize_3x4": _resize((9, 5, 32), (27, 20)),
    "model_resize_5x5": _resize((3, 4, 12), (15, 20)),
}
