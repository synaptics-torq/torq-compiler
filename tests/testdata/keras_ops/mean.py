"""Keras test models: GlobalAveragePooling2D (TFLite MEAN)."""

import tensorflow as tf

from torq.testing.keras_builder import functional, layers, sequential

CASES = {
    "model180_mean_12x8x256_to_1x256": sequential((12, 8, 256), layers.GlobalAveragePooling2D()),
    "model181_mean_inp_4x8x6_zp_64": sequential((4, 8, 6), layers.GlobalAveragePooling2D()),
    "model182_mean_inp_22x32x16_zp_1": sequential((23, 32, 16), layers.GlobalAveragePooling2D()),
    "model183_mean_inp_37x17x11_zp_minus127": sequential(
        (37, 17, 11),
        layers.GlobalAveragePooling2D(),
    ),
    "model184_mean_inp_12x16_64_zp_minus76": sequential(
        (12, 16, 64),
        layers.GlobalAveragePooling2D(),
    ),
    "model185_mean_inp_32x1x512_zp_74_scale_3": sequential(
        (32, 1, 512),
        layers.GlobalAveragePooling2D(),
    ),
    "model186_mean_53x37x1_zp_minus25_scale_39": sequential(
        (53, 37, 1),
        layers.GlobalAveragePooling2D(),
    ),
    "model187_mean_1x32x512_zp_26": sequential((1, 32, 512), layers.GlobalAveragePooling2D()),
    "model188_mean_11x1x1_zp_52_scale_em7": sequential((11, 1, 1), layers.GlobalAveragePooling2D()),
    # The functional form: model.layers has the InputLayer.
    "model189_mean_1x173x1_zp_minus71_scale19": functional(
        (1, 173, 1), lambda x: tf.keras.layers.GlobalAveragePooling2D()(x),
    ),
}
