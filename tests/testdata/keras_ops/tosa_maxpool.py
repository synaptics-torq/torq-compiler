"""Keras test models: max pools ported from tests/testdata/tosa_ops.

Each case keeps the name of the MLIR file it replaces and its quantization. The pads are those of
Keras "same" (the extra row or column at the end) or "valid"; MLIR files with other pads stay in
tests/testdata/tosa_ops.
"""

import tensorflow as tf

from torq.testing.keras_builder import functional, layers, sequential

QUANTIZE = "int8"

# As test_tosa_ops does for its cases: they must run on the slices.
DEFAULT_CONFIG = {"compiler_options": ["--torq-disable-host", "--torq-disable-css"]}


def _maxpool(shape, pool_size, strides, padding="valid"):
    return sequential(shape, layers.MaxPooling2D(pool_size=pool_size, strides=strides,
                                                 padding=padding))


def _maxpools(shape, *pools, concat=False):
    """Several max pools of the same input, each (pool_size, strides, padding); with concat, the
    input and the pools concatenated along the channels (SPP), else one output per pool."""
    def build(x):
        L = tf.keras.layers
        ys = [L.MaxPooling2D(pool_size=k, strides=s, padding=p)(x) for k, s, p in pools]
        return L.Concatenate(axis=-1)([x, *ys]) if concat else ys
    return functional(shape, build)


def _padded_maxpool(shape, pad, pool_size, strides):
    """A VALID max pool after a ZeroPadding2D of ((top, bottom), (left, right)). The inputs are
    calibrated on [0, 1), so a zero pad is the lowest value and the pad folds into the pool."""
    return functional(shape, lambda x: tf.keras.layers.MaxPooling2D(
        pool_size=pool_size, strides=strides)(tf.keras.layers.ZeroPadding2D(pad)(x)))


# Not ported: maxpool2d-stride1-k3x3-32x32x4 and maxpool_int16_batch1_3x3_stride1x1_16x4
# (test_tile_fit_probe uses their MLIR files).
CASES = {
    "maxpool-seg-tile-1": _maxpool((28, 28, 512), 2, 2),
    "maxpool-seg-tile": _maxpool((56, 56, 256), 2, 2),
    "maxpool1d_i16_stride4_k4x1_256x16x32": _maxpool((256, 16, 32), (4, 1), (4, 1)),
    "maxpool1d_i8_stride4_k4x1_256x16x32": _maxpool((256, 16, 32), (4, 1), (4, 1)),
    "maxpool2d-stride1-k3x3-32x32x4-i16": _maxpool((32, 32, 4), 3, 1, "same"),
    "maxpool2d-stride2-k2x2-32x32x4": _maxpool((32, 32, 4), 2, 2),
    "maxpool2d-stride2-k2x2-8x8x1": _maxpool((8, 8, 1), 2, 2),
    "maxpool2d-stride2-k2x2-8x8x32": _maxpool((8, 8, 32), 2, 2),
    "maxpool2d-stride2-k2x2-8x8x4": _maxpool((8, 8, 4), 2, 2),
    "maxpool2d-stride2-k3x3-32x32x4": _maxpool((32, 32, 4), 3, 2, "same"),
    "maxpool2d-stride2-k3x3-8x8x4": _maxpool((8, 8, 4), 3, 2, "same"),
    "maxpool2d-stride2-k3x3-bottom-right-pad": _maxpool((8, 8, 16), 3, 2, "same"),
    "maxpool2d-stride2-k3x3-pad-224x224x64": _maxpool((224, 224, 64), 2, 2),
    "maxpool2d_i16_s2_k3x3_32x32x4": _maxpool((32, 32, 1), 3, 2, "same"),
    "maxpool2d_i16_s2_k7x7_10x10x4": _maxpool((10, 10, 4), 7, 2, "same"),
    "maxpool2d_int8_batch1_2x2_stride2x2_32x16x4": _maxpool((16, 4, 32), 2, 2),
    "maxpool_f11_valid_24x24x8": _maxpool((24, 24, 8), 11, 1),
    "maxpool_f13_20x20x8": _maxpool((20, 20, 8), 13, 1, "same"),
    "maxpool_f5_15x20x384": _maxpool((15, 20, 384), 5, 1, "same"),
    "maxpool_f5_8x8x4": _maxpool((8, 8, 4), 5, 1, "same"),
    "maxpool_f5_s2_10x10x4": _maxpool((10, 10, 4), 5, 2, "same"),
    "maxpool_f7_8x8x4": _maxpool((8, 8, 4), 7, 1, "same"),
    "maxpool_f7_s2_10x10x4": _maxpool((10, 10, 4), 7, 2, "same"),
    "maxpool_f8_s2_pad3x4_21x21x8": _maxpool((21, 21, 8), 8, 2, "same"),
    "maxpool_f9_20x20x8": _maxpool((20, 20, 8), 9, 1, "same"),
    "maxpool_f9_s2_21x21x8": _maxpool((21, 21, 8), 9, 2, "same"),
    "maxpool_int8_batch1_3x3_stride1x1_16x4": _maxpool((16, 4, 32), 3, 1, "same"),
    "maxpool_k1_s2_21x21x16": _maxpool((21, 21, 16), 1, 2),
    "maxpool_k1x2_s1x2_2x32x16": _maxpool((2, 32, 16), (1, 2), (1, 2)),
    "maxpool_k1x2_s1x2_64x64x16": _maxpool((64, 64, 16), (1, 2), (1, 2)),
    "maxpool_k1x8_s1x8_16x256x64": _maxpool((16, 256, 64), (1, 8), (1, 8)),
    "maxpool_k2_s1_pad01_16x16x8": _maxpool((16, 16, 8), 2, 1, "same"),
    "maxpool_k2_s4_22x22x8": _maxpool((22, 22, 8), 2, 4),
    "maxpool_k2x3_s2_pad0111_11x11x8": _maxpool((11, 11, 8), (2, 3), 2, "same"),
    "maxpool_k3_s3_18x18x8": _maxpool((18, 18, 8), 3, 3),
    "maxpool_k3x2_s2_pad1101_11x11x8": _maxpool((11, 11, 8), (3, 2), 2, "same"),
    "maxpool_k3x2_s2x1_17x16x8": _maxpool((17, 16, 8), (3, 2), (2, 1)),
    "maxpool_k3x5_s2_same_11x11x8": _maxpool((11, 11, 8), (3, 5), 2, "same"),
    # From synaptics-torq/torq-compiler-dev#2743: stride 2, H VALID and W padded with the W kernel
    # centre, so the stride offset must come from the centre of each axis.
    "maxpool_k3x5_s2_pad_w_only_9x41x8": _padded_maxpool((9, 41, 8), ((0, 0), (2, 2)), (3, 5), 2),
    "maxpool_k4_s1_16x16x8": _maxpool((16, 16, 8), 4, 1),
    "maxpool_k4_s1_pad12_16x16x8": _maxpool((16, 16, 8), 4, 1, "same"),
    "maxpool_k4_s2_20x20x8": _maxpool((20, 20, 8), 4, 2),
    "maxpool_k4_s2_pad12_21x21x8": _maxpool((21, 21, 8), 4, 2, "same"),
    "maxpool_k4_s4_20x20x8": _maxpool((20, 20, 8), 4, 4),
    "maxpool_k4x6_s2_20x20x8": _maxpool((20, 20, 8), (4, 6), 2),
    "maxpool_k5_s3_17x17x8": _maxpool((17, 17, 8), 5, 3),
    "maxpool_k5_s4_pad2_21x21x8": _maxpool((21, 21, 8), 5, 4, "same"),
    "maxpool_k5x3_s2_11x11x8": _maxpool((11, 11, 8), (5, 3), 2),
    "maxpool_k6_s2_20x20x8": _maxpool((20, 20, 8), 6, 2),
    "maxpool_k6_s2_pad23_21x21x8": _maxpool((21, 21, 8), 6, 2, "same"),
    "maxpool_k9_s3_21x21x8": _maxpool((21, 21, 8), 9, 3),
    "maxpool_k9_s4_pad4_21x21x8": _maxpool((21, 21, 8), 9, 4, "same"),
    "maxpool_k9x3_20x20x8": _maxpool((20, 20, 8), (9, 3), 1, "same"),
    "maxpool_noshare_pad_k3_k5_20x20x8": _maxpools((20, 20, 8), (3, 1, "same"), (5, 1, "valid")),
    "maxpool_share_k3s1_k5s2_21x21x8": _maxpools((21, 21, 8), (3, 1, "same"), (5, 2, "same")),
    "maxpool_share_valid_k3_k7_20x20x8": _maxpools((20, 20, 8), (3, 1, "valid"), (7, 1, "valid")),
    "maxpool_spp_5_9_13_20x20x16": _maxpools(
        (20, 20, 16), (5, 1, "same"), (9, 1, "same"), (13, 1, "same"), concat=True,
    ),
}

CONFIG = {name: {"quantize": "int16"} for name in (
    "maxpool1d_i16_stride4_k4x1_256x16x32",
    "maxpool2d-stride1-k3x3-32x32x4-i16",
    "maxpool2d_i16_s2_k3x3_32x32x4",
    "maxpool2d_i16_s2_k7x7_10x10x4",
)}
