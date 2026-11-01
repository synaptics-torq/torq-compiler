"""Keras test models: transposes, depth-to-space and space-to-depth ported from
tests/testdata/tosa_ops.

Each case keeps the name of the MLIR file it replaces and its quantization. The first dimension of
each input is the batch.
"""

import tensorflow as tf

from torq.testing.keras_builder import functional

QUANTIZE = "int8"

# As test_tosa_ops does for its cases: they must run on the slices.
DEFAULT_CONFIG = {"compiler_options": ["--torq-disable-host", "--torq-disable-css"]}


def _transpose(shape, perm):
    return functional(shape[1:], lambda x: tf.keras.layers.Lambda(
        lambda t: tf.transpose(t, perm))(x), batch_size=shape[0])


def _blocks(shape, split, perm, merged):
    """reshape to split, transpose by perm, reshape to merged: the TOSA form of depth-to-space and
    space-to-depth (TFLite DEPTH_TO_SPACE has no int16 quantization)."""
    def blocks(t):
        return tf.reshape(tf.transpose(tf.reshape(t, split), perm), merged)
    return functional(shape, lambda x: tf.keras.layers.Lambda(blocks)(x))


def _depth_to_space(shape, crd=False):
    """Depth-to-space with blocks of 2, DCR or CRD order."""
    h, w, c = shape
    if crd:
        split, perm = (1, h, w, c // 4, 2, 2), (0, 1, 4, 2, 5, 3)
    else:
        split, perm = (1, h, w, 2, 2, c // 4), (0, 1, 3, 2, 4, 5)
    return _blocks(shape, split, perm, (1, 2 * h, 2 * w, c // 4))


def _space_to_depth(shape):
    """Space-to-depth with blocks of 2."""
    h, w, c = shape
    return _blocks(shape, (1, h // 2, 2, w // 2, 2, c), (0, 1, 3, 2, 4, 5),
                   (1, h // 2, w // 2, 4 * c))


# Not ported: d2s_8x8x128xi8_crd (test_fold_reshape_into_binding uses its MLIR file).
CASES = {
    "transpose-02341-2x64x4x8x3xi8": _transpose((2, 64, 4, 8, 3), (0, 2, 3, 4, 1)),
    "transpose-120-64x32x33xi8": _transpose((64, 32, 33), (1, 2, 0)),
    "transpose-201-33x40x64xi8": _transpose((33, 40, 64), (2, 0, 1)),
    "transpose-2dim-aligned": _transpose((128, 64), (1, 0)),
    "transpose-2dim": _transpose((128, 49), (1, 0)),
    "transpose-3dim": _transpose((128, 49, 2), (1, 2, 0)),
    "transpose-3dim_1": _transpose((1, 8, 4), (0, 2, 1)),
    "transpose-3dim_2": _transpose((1, 256, 4), (0, 2, 1)),
    "transpose-5dim": _transpose((128, 49, 2, 2, 3), (1, 2, 3, 4, 0)),
    "transpose-5dim_1": _transpose((1, 1, 64, 32, 32), (0, 1, 4, 2, 3)),
    "transpose-5dim_2": _transpose((1, 1, 64, 2, 36), (0, 1, 3, 4, 2)),
    "transpose-5dim_3": _transpose((1, 2, 64, 2, 36), (0, 1, 3, 4, 2)),
    "transpose-64x64xi8": _transpose((64, 64), (1, 0)),
    "transpose-95x159xi8": _transpose((95, 159), (1, 0)),
    "transpose-96x160xi16": _transpose((96, 160), (1, 0)),
    "transpose-96x160xi8": _transpose((96, 160), (1, 0)),
    "transpose-nchw2nhwc-1x64x32x32xi8": _transpose((1, 64, 32, 32), (0, 2, 3, 1)),
    "transpose-nchw2nhwc-2x48x7x9xi8": _transpose((2, 48, 7, 9), (0, 2, 3, 1)),
    "transpose-nhwc2nchw-1x2x30x64xi8": _transpose((1, 2, 30, 64), (0, 3, 1, 2)),
    "transpose-nhwc2nchw-1x2x31x64xi8": _transpose((1, 2, 31, 64), (0, 3, 1, 2)),
    "transpose-nhwc2nchw-1x2x32x64xi8": _transpose((1, 2, 32, 64), (0, 3, 1, 2)),
    "transpose-nhwc2nchw-1x2x33x64xi8": _transpose((1, 2, 33, 64), (0, 3, 1, 2)),
    "transpose-nhwc2nchw-1x32x32x64xi8": _transpose((1, 32, 32, 64), (0, 3, 1, 2)),
    "transpose-nhwc2nchw-1x3x21x64xi8": _transpose((1, 3, 21, 64), (0, 3, 1, 2)),
    "transpose-nhwc2nchw-1x5x13x64xi8": _transpose((1, 5, 13, 64), (0, 3, 1, 2)),
    "transpose-nhwc2nchw-2x7x9x48xi8": _transpose((2, 7, 9, 48), (0, 3, 1, 2)),
    "transpose": _transpose((1, 1280, 7, 7), (0, 2, 3, 1)),
    "d2s_16x16x128xi16_crd": _depth_to_space((16, 16, 128), crd=True),
    "d2s_16x16x128xi16_dcr": _depth_to_space((16, 16, 128)),
    "d2s_16x16x128xi8_crd": _depth_to_space((16, 16, 128), crd=True),
    "d2s_16x16x128xi8_dcr": _depth_to_space((16, 16, 128)),
    "d2s_8x8x128xi16_crd": _depth_to_space((8, 8, 128), crd=True),
    "d2s_8x8x128xi16_dcr": _depth_to_space((8, 8, 128)),
    "d2s_8x8x128xi8_dcr": _depth_to_space((8, 8, 128)),
    "d2s_8x8x4xi16": _depth_to_space((8, 8, 4)),
    "d2s_8x8x4xi8": _depth_to_space((8, 8, 4)),
    "d2s_9x9x128xi16_dcr": _depth_to_space((9, 9, 128)),
    "s2d_16x16x1xi8": _space_to_depth((16, 16, 1)),
}

CONFIG = {name: {"quantize": "int16"} for name in (
    "transpose-96x160xi16",
    "d2s_16x16x128xi16_crd",
    "d2s_16x16x128xi16_dcr",
    "d2s_8x8x128xi16_crd",
    "d2s_8x8x128xi16_dcr",
    "d2s_8x8x4xi16",
    "d2s_9x9x128xi16_dcr",
)}
