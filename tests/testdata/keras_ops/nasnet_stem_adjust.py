"""Keras test model: the NASNetMobile stem and the adjust block of its second stem cell.

The stem conv output (111x111x32) feeds two branches: a strided 1x1 average pool, and a
ZeroPadding2D + Cropping2D + strided 1x1 average pool. Each branch ends with a 1x1 conv, and the
branches are concatenated. On SL2610, the stem output (394,272 bytes) only fits in LRAM if
tile-and-fuse also tiles the fuse-group producer that the padding branch keeps alive.
"""

import tensorflow as tf

from torq.testing.keras_builder import functional

L = tf.keras.layers


def _stem_adjust(x):
    """NASNetMobile stem conv, then the adjust block of cell stem_2 (filters 22)."""
    p = L.ReLU()(L.Conv2D(32, 3, strides=2, use_bias=False)(x))
    p1 = L.AveragePooling2D(1, strides=2)(p)
    p1 = L.Conv2D(11, 1, use_bias=False)(p1)
    p2 = L.ZeroPadding2D(((0, 1), (0, 1)))(p)
    p2 = L.Cropping2D(((1, 0), (1, 0)))(p2)
    p2 = L.AveragePooling2D(1, strides=2)(p2)
    p2 = L.Conv2D(11, 1, use_bias=False)(p2)
    return L.Concatenate()([p1, p2])


CASES = {"nasnet_stem_adjust_224x224x3": functional((224, 224, 3), _stem_adjust)}
QUANTIZE = "int8"
