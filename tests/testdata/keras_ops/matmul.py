"""Keras test models: matmul of two activations, as in an attention block."""

import tensorflow as tf

from torq.testing.keras_builder import functional

L = tf.keras.layers


def _matmul(transpose_b):
    return L.Lambda(lambda t: tf.matmul(t[0], t[1], transpose_b=transpose_b))


def _scores(shape, channels):
    """q @ k^T, with q and k two Dense projections of the input."""
    return functional(shape, lambda x: _matmul(True)([L.Dense(channels)(x),
                                                      L.Dense(channels)(x)]))


def _attention_values(shape, keys, channels):
    """a @ v; the ReLU gives a the zero point -128 of a softmax output."""
    return functional(shape, lambda x: _matmul(False)([L.ReLU()(L.Dense(keys)(x)),
                                                       L.Dense(channels)(x)]))


CASES = {
    "matmul_qkT_inp4x100x32": _scores((4, 100, 32), 32),
    "matmul_av_inp4x100x64": _attention_values((4, 100, 32), 100, 64),
    "matmul_qkT_inp2x7x9": _scores((2, 7, 9), 9),
}

# Both operands are activations, so the int8 matmul has a zero point on each of them.
DEFAULT_CONFIG = {"quantize": ("int8",)}
