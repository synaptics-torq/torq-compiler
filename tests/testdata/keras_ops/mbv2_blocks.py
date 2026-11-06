"""Keras test models: blocks 1 and 2 of MobileNetV2, with the residual add.

Block 1 (expand to 96, stride-2 depthwise, project to 24) makes the residual. Block 2 (expand to
144, depthwise, project to 24) adds its result to it. On SL2610, tile-and-fuse tiles block 1 on
rows in a loop with peeled first and last tiles. It tiles block 2 on rows in tiles that it
unrolls. tests/compiler/test_lram_loads.py checks the LRAM plan of these tiles.
"""

import tensorflow as tf

from torq.testing.keras_builder import functional

L = tf.keras.layers


def _blocks(x):
    """Blocks 1 and 2 of MobileNetV2, then the residual add."""
    y = L.ReLU(6.0)(L.Conv2D(96, 1)(x))
    y = L.ReLU(6.0)(L.DepthwiseConv2D(3, strides=2, padding="same")(y))
    r = L.Conv2D(24, 1)(y)
    y = L.ReLU(6.0)(L.Conv2D(144, 1)(r))
    y = L.ReLU(6.0)(L.DepthwiseConv2D(3, padding="same")(y))
    return L.Add()([r, L.Conv2D(24, 1)(y)])


CASES = {"mbv2_blocks_1_2_112x112x16": functional((112, 112, 16), _blocks)}
QUANTIZE = "int8"
