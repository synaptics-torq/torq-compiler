"""Keras test models: resize, concat, gather, mean, reshape and a sigmoid-gated mul ported
from tests/testdata/tosa_ops.

Each case keeps the name of the MLIR file it replaces and its quantization.
"""

import tensorflow as tf

from torq.testing.keras_builder import functional, layers, sequential

QUANTIZE = "int8"

# As test_tosa_ops does for its cases: they must run on the slices.
DEFAULT_CONFIG = {"compiler_options": ["--torq-disable-host", "--torq-disable-css"]}

L = tf.keras.layers


def _resize(shape, size):
    """Nearest-neighbor resize with half-pixel centers (TOSA scale 2 * size / 2 * input)."""
    return functional(shape, lambda x: L.Lambda(
        lambda t: tf.image.resize(t, size, method="nearest"))(x))


def _concat(shapes, axis):
    """Concatenation of inputs that all have the batch size 1."""
    return functional(list(shapes), lambda *xs: L.Lambda(lambda ts: tf.concat(ts, axis))(list(xs)))


def _gather(shape, indices):
    """Gather along axis 1 with constant indices."""
    return functional(shape, lambda x: L.Lambda(lambda t: tf.gather(t, indices, axis=1))(x))


# Not ported: gather-i16 (test_tosa_ops gives it a longer compile budget).
CASES = {
    "resize-31x31x33xi8": _resize((31, 31, 33), (62, 62)),
    "resize-4x4x32xi8": _resize((4, 4, 32), (8, 8)),
    "resize-8x8x32xi8": _resize((8, 8, 32), (16, 16)),
    "concat-a": _concat([(21,), (21,)], 0),
    "concat-b": _concat([(512, 4), (256, 4), (96, 4)], 1),
    "gather": _gather((256, 32), [200, 100, 30, 23, 45, 65, 76, 78, 90, 10, 22, 55, 15, 128, 222,
                                  202]),
    "mean_hw-i8": sequential((7, 7, 96), layers.GlobalAveragePooling2D()),
    "reducesum-avgpool": sequential((7, 7, 1280), layers.GlobalAveragePooling2D()),
    "reshape": sequential((7, 7, 512), layers.Reshape((1, 1, 25088))),
    # a * sigmoid(b), from the EfficientNet stem: a table (LOGISTIC) then a mul
    "efficientnet-sigmoid-rescale-mul": functional(
        [(112, 112, 32), (112, 112, 32)],
        lambda a, b: L.Multiply()([a, L.Activation("sigmoid")(b)]),
    ),
}
