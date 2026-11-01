"""Keras test models: argmax ported from tests/testdata/tosa_ops.

Each case keeps the name of the MLIR file it replaces. The first dimension of each input is the
batch; the result is int32.
"""

import tensorflow as tf

from torq.testing.keras_builder import functional

QUANTIZE = "int8"

# As test_tosa_ops does for its cases: they must run on the slices.
DEFAULT_CONFIG = {"compiler_options": ["--torq-disable-host", "--torq-disable-css"]}


def _argmax(shape, axis):
    return functional(shape[1:], lambda x: tf.keras.layers.Lambda(
        lambda t: tf.math.argmax(t, axis=axis, output_type=tf.int32))(x), batch_size=shape[0])


CASES = {
    "argmax-1x10x30-1axis": _argmax((1, 10, 30), 1),
    "argmax-1x128x7-2axis": _argmax((1, 128, 7), 2),
    "argmax-1x256x50-2axis": _argmax((1, 256, 50), 2),
    "argmax-1x300x64-1axis": _argmax((1, 300, 64), 1),
    "argmax-1x4x64-1axis": _argmax((1, 4, 64), 1),
    "argmax-1x50x256-1axis": _argmax((1, 50, 256), 1),
    "argmax-1x64x4-2axis": _argmax((1, 64, 4), 2),
    "argmax-1x64x8-2axis": _argmax((1, 64, 8), 2),
    "argmax-1x7x128-1axis": _argmax((1, 7, 128), 1),
    "argmax-1x8x64-1axis": _argmax((1, 8, 64), 1),
    "argmax-2x16x80-1axis": _argmax((2, 16, 80), 1),
    "argmax-64x8-0axis": _argmax((64, 8), 0),
    "argmax-6x4x40-0axis": _argmax((6, 4, 40), 0),
    "argmax-8x24-1axis": _argmax((8, 24), 1),
}
