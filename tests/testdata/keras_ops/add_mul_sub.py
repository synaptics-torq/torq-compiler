"""Keras test models: Add, Multiply and Subtract, of tensors and of constants."""

import tensorflow as tf

from torq.testing.keras_builder import conv2d, functional, layers, sequential

L = tf.keras.layers


def _conv(x, filters=1, kernel_size=(3, 3), relu=False):
    y = L.Conv2D(filters=filters, kernel_size=kernel_size, padding="same")(x)
    return L.ReLU()(y) if relu else y


def _conv_relu(shape, *kernel_sizes):
    """A chain of 1-filter same-padded Conv2D + ReLU, one per kernel size."""
    specs = []
    for kernel_size in kernel_sizes:
        specs += [layers.Conv2D(filters=1, kernel_size=kernel_size, padding="same"), layers.ReLU()]
    return sequential(shape, *specs)


def _with_conv(shape, merge, *, conv_first=False, **conv):
    """merge([inputs, conv(inputs)]), or merge([conv(inputs), inputs]) with conv_first."""
    def build(x):
        y = _conv(x, **conv)
        return getattr(L, merge)()([y, x] if conv_first else [x, y])
    return functional(shape, build)


def _two_convs(shape, merge, relu):
    """merge([conv(inputs), conv(inputs)]) with two different convs."""
    return functional(shape, lambda x: getattr(L, merge)()([_conv(x, relu=relu),
                                                            _conv(x, relu=relu)]))


def _self(shape, *merges):
    """Each merge in turn applied to [x, x]."""
    def build(x):
        for merge in merges:
            x = getattr(L, merge)()([x, x])
        return x
    return functional(shape, build)


def _lambda(shape, fn):
    return functional(shape, lambda x: L.Lambda(fn)(x))


class SubtractConstant(L.Layer):
    """x - c, with c a scalar weight."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.constant = self.add_weight(name="sub_constant", shape=(),
                                        initializer=tf.keras.initializers.RandomNormal(seed=42),
                                        trainable=True)

    def call(self, inputs):
        return tf.subtract(inputs, self.constant)


CASES = {
    "model050_mult_inp1x10x10x4": _with_conv((10, 10, 4), "Multiply", filters=4,
                                             kernel_size=(4, 4)),
    "model051_mult_inp1x4x4x1": _with_conv((4, 4, 1), "Multiply"),
    "model052_mult_inp1x4x4x1_zp128_AB": _with_conv((4, 4, 1), "Multiply", relu=True),
    "model053_mult_inp1x4x4x1_zp128_B": _with_conv((4, 4, 1), "Multiply", relu=True),
    "model054_mult_inp1x4x4x1_zp128_A": _with_conv((4, 4, 2), "Multiply", filters=2,
                                                   kernel_size=(2, 2)),
    "model060_add_inp1x4x4x1": _with_conv((4, 4, 1), "Add"),
    "model061_add_inp1x4x4x1_zp128_x1": _with_conv((4, 4, 1), "Add", relu=True),
    "model062_add_inp1x4x4x1_zp128_x2": _with_conv((4, 4, 1), "Add", relu=True),
    "model120_mult_inp1x8x8x1": _self((8, 8, 1), "Multiply"),
    "model121_add_inp1x8x8x1": _self((8, 8, 1), "Add"),
    "model122_conv_mult_inp1x8x8x1": _self((8, 8, 1), "Multiply"),
    "model123_conv_add_inp1x8x8x1": _with_conv((8, 8, 1), "Add", relu=True),
    "model130_conv3x3_inp1x3x3x1": _conv_relu((3, 3, 1), (3, 3)),
    "model131_mult_inp1x3x3x1": _self((3, 3, 1), "Multiply"),
    "model132_add_inp1x3x3x1": _self((3, 3, 1), "Add"),
    "model133_conv3x3_mult_inp1x3x3x1": _with_conv((3, 3, 1), "Multiply", relu=True),
    "model134_conv3x3_add_inp1x3x3x1": _with_conv((3, 3, 1), "Add", relu=True),
    "model135_conv3x3_conv2x2_inp1x3x3x1": _conv_relu((3, 3, 1), (3, 3), (2, 2)),
    "model136_conv2x2_inp1x3x3x1": _conv_relu((3, 3, 1), (2, 2)),
    "model137_add_mult_inp1x3x3x1": _self((3, 3, 1), "Add", "Multiply"),
    "model138_mult_mult_inp1x3x3x1": _self((3, 3, 1), "Multiply", "Multiply"),
    "model140_conv3x3_inp1x4x4x1": _conv_relu((4, 4, 1), (3, 3)),
    "model141_mult_inp1x4x4x1": _self((4, 4, 1), "Multiply"),
    "model142_add3x3_inp1x4x4x1": _self((4, 4, 1), "Add"),
    "model143_conv3x3_mult_inp1x4x4x1": _with_conv((4, 4, 1), "Multiply", relu=True),
    "model144_conv3x3_add_inp1x4x4x1": _with_conv((4, 4, 1), "Add", relu=True),
    "model145_conv3x3_conv2x2_inp1x4x4x1": _conv_relu((4, 4, 1), (3, 3), (2, 2)),
    "model146_conv2x2_inp1x4x4x1": _conv_relu((4, 4, 1), (2, 2)),
    "model147_conv_inp1x32x32x16_16x3x3_same_stride1x1": conv2d((32, 32, 16), 16, (3, 3), "same"),
    "model150_sub_1x6x8x4_zp128_B": _with_conv((6, 8, 4), "Subtract", relu=True),
    "model151_sub_1x6x8x4_zp128_A": _with_conv((6, 8, 4), "Subtract", relu=True,
                                               conv_first=True),
    "model152_sub_1x6x8x4_zp128_AB": _two_convs((6, 8, 4), "Subtract", relu=True),
    "model153_sub_1x6x8x4_zp128_none": _two_convs((6, 8, 4), "Subtract", relu=False),
    "model154_sub_s2v_1x6x8x4": _lambda((6, 8, 4), lambda x: tf.constant(0.0) - x),
    "model155_sub_v2s_1x6x8x4": functional((6, 8, 4), lambda x: SubtractConstant()(x)),
    "model156_add_s2v_1x6x8x4": _lambda((6, 8, 4), lambda t: t + 1.0),
    "model157_add_v2s_1x6x8x4": _lambda((6, 8, 4), lambda t: t + 1.0),
    "model158_mul_s2v_1x6x8x4": _lambda((6, 8, 4), lambda t: t * 2.0),
    "model159_mul_v2s_1x6x8x4": _lambda((6, 8, 4), lambda t: t * 2.0),
    "model470_add16x8_positive_s2v": functional((10, 10, 3), lambda x: x + 1.0),
    "model471_add16x8_negative_s2v": _lambda((4, 1, 1), lambda t: t + -1.0),
    "model472_mult16x8_neg_s2v": _lambda((10, 10, 3), lambda x: x * -2.0),
    "model473_mult16x8_pos_v2s": _lambda((10, 10, 3), lambda x: x * 2.0),
    "model474_sub16x8_pos_v2s": _lambda((10, 10, 3), lambda t: t - 1.0),
    "model475_sub16x8_neg_v2s": _lambda((10, 10, 3), lambda x: x - 1.0),
}
