"""Keras test models: Conv2DTranspose and Conv1DTranspose.

Each case name says what it tests: kernel, strides, padding, input shape (H x W x C), output
channels and nobias for convs without bias, e.g. conv2d_transpose_k1x3_s1x2_same_1x4x1_oc1.
"""

import tensorflow as tf

from torq.testing.keras_builder import conv2d_transpose, functional, layers, sequential

# (input H, W, C), output channels, kernel (h, w), strides (h, w), padding
_WITH_BIAS = [
    ((1, 4, 1), 1, (1, 3), (1, 1), "same"),
    ((1, 4, 1), 1, (1, 3), (1, 2), "same"),
    ((1, 4, 1), 2, (1, 3), (1, 1), "same"),
    ((1, 4, 1), 1, (1, 3), (1, 1), "valid"),
    ((1, 4, 1), 1, (1, 3), (1, 2), "valid"),
    ((1, 4, 1), 2, (1, 3), (1, 1), "valid"),
    ((3, 3, 1), 1, (3, 3), (1, 1), "same"),
    ((3, 3, 1), 1, (3, 3), (2, 2), "same"),
    ((4, 4, 2), 3, (2, 2), (1, 1), "same"),
    ((5, 5, 3), 2, (4, 4), (1, 1), "valid"),
    ((7, 7, 2), 3, (4, 3), (2, 2), "same"),
    ((1, 4, 16), 16, (1, 3), (1, 1), "same"),
    ((1, 4, 16), 16, (1, 3), (1, 1), "valid"),
    ((1, 4, 8), 8, (1, 3), (1, 2), "same"),
    ((1, 4, 8), 8, (1, 3), (1, 2), "valid"),
    ((1, 4, 2), 1, (1, 3), (1, 1), "same"),
    ((1, 4, 2), 1, (1, 3), (1, 2), "same"),
    ((1, 4, 2), 1, (1, 3), (1, 1), "valid"),
    ((1, 4, 2), 1, (1, 3), (1, 2), "valid"),
    ((3, 3, 1), 1, (3, 3), (1, 1), "valid"),
    ((3, 3, 1), 1, (3, 3), (2, 2), "valid"),
    ((4, 4, 2), 3, (3, 3), (2, 2), "valid"),
    ((3, 3, 1), 1, (3, 3), (3, 3), "same"),
    ((3, 3, 1), 1, (3, 3), (3, 3), "valid"),
    ((1, 64, 32), 32, (1, 3), (1, 1), "same"),
    ((5, 5, 3), 23, (4, 4), (1, 1), "valid"),
    ((11, 10, 7), 11, (2, 4), (2, 4), "valid"),
    ((1, 8, 8), 8, (1, 5), (1, 2), "same"),
]

_NO_BIAS = [
    ((1, 32, 48), 48, (1, 3), (1, 2), "same"),
    ((1, 64, 32), 32, (1, 3), (1, 1), "same"),
    ((1, 4, 1), 1, (1, 3), (1, 1), "same"),
    ((1, 4, 1), 1, (1, 3), (1, 2), "same"),
    ((1, 4, 2), 2, (1, 3), (1, 1), "same"),
]


def _name(shape, filters, kernel, strides, padding, after=""):
    h, w, c = shape
    return (f"conv2d_transpose_k{kernel[0]}x{kernel[1]}_s{strides[0]}x{strides[1]}_{padding}_"
            f"{h}x{w}x{c}_oc{filters}{after}")


def _no_bias(shape, filters, kernel, strides, padding):
    return sequential(shape, layers.Conv2DTranspose(
        filters=filters, kernel_size=kernel, strides=strides, padding=padding, use_bias=False))


CASES = {}


def _add(name, builder):
    if name in CASES:
        raise ValueError(f"conv_transpose: {name} is listed twice")
    CASES[name] = builder


for _c in _WITH_BIAS:
    _add(_name(*_c), conv2d_transpose(*_c))
for _c in _NO_BIAS:
    _add(_name(*_c, "_nobias"), _no_bias(*_c))
_add("conv1d_transpose_k3_s2_same_12x4_oc4_nobias", sequential(
    (12, 4),
    layers.Conv1DTranspose(filters=4, kernel_size=3, strides=2, padding="same", use_bias=False),
))
_add("conv2d_transpose_k4x4_s2x2_same_8x8x4_oc4_seed21321", sequential(
    (8, 8, 4),
    layers.Conv2DTranspose(filters=4, kernel_size=4, strides=2, padding="same"),
    seed=21321,
))


def _shared_input_convt(x):
    """A 1x1 conv whose output feeds a stride (1, 2) 1x3 ConvT and another 1x1 conv."""
    y = tf.keras.layers.Conv2D(8, 1)(x)
    return [tf.keras.layers.Conv2DTranspose(8, (1, 3), strides=(1, 2), padding="same")(y),
            tf.keras.layers.Conv2D(8, 1)(y)]


_add("conv2d_transpose_k1x3_s1x2_same_1x8x8_oc8_shared_input",
     functional((1, 8, 8), _shared_input_convt))
