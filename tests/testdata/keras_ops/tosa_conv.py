"""Keras test models: convolutions, depthwise convolutions and fully connected layers ported from
tests/testdata/tosa_ops.

Each case keeps the name of the MLIR file it replaces and its quantization. The pads are those of
Keras "same" or "valid", and the biases are random where the MLIR file had a non-zero bias. The
inputs are calibrated on [0, 1), so their int8 zero point is -128, as in most of the MLIR files.
Not ported: a separate pad before a conv (TFLite folds ZeroPadding2D into the conv) and int16
depthwise convs (the TFLite converter fails to quantize them).
"""

import tensorflow as tf

from torq.testing.keras_builder import functional, layers, sequential

QUANTIZE = "int8"

# As test_tosa_ops does for its cases: they must run on the slices.
DEFAULT_CONFIG = {"compiler_options": ["--torq-disable-host", "--torq-disable-css"]}

_BIAS = {"bias_initializer": "random_uniform"}


def _conv(shape, filters, kernel_size, strides, padding="valid", bias=True, relu6=False):
    specs = [layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                           padding=padding, **(_BIAS if bias else {}))]
    return sequential(shape, *specs, *([layers.ReLU(6.0)] if relu6 else []))


def _depthwise(shape, kernel_size, strides, padding="valid", bias=True):
    return sequential(shape, layers.DepthwiseConv2D(kernel_size=kernel_size, strides=strides,
                                                    padding=padding, **(_BIAS if bias else {})))


def _dense(batch, inputs, units):
    return sequential((inputs,), layers.Dense(units, **_BIAS), batch_size=batch)


def _pool_then_dense(shape, units):
    """Global average pooling (TFLite MEAN) then a fully connected layer, as a classifier head."""
    return functional(shape, lambda x: tf.keras.layers.Dense(units, **_BIAS)(
        tf.keras.layers.GlobalAveragePooling2D()(x)))


# Not ported: dw_peeling (test_torqrt uses its MLIR file).
CASES = {
    "conv2d_3x5_stride2_same_11x11x4": _conv((11, 11, 4), 8, (3, 5), 2, "same"),
    "conv2d_4x4_stride2_valid_12x12x4": _conv((12, 12, 4), 8, 4, 2),
    "conv_5x5_odd_frame_stride2": _conv((5, 5, 128), 256, 3, 2, "same", bias=False),
    "conv_s2_rgb_input_128x128x3-i8": _conv((128, 128, 3), 32, 3, 2, "same", relu6=True),
    "conv_s2_rgb_input_224x224x3-i8": _conv((224, 224, 3), 32, 3, 2, "same", relu6=True),
    "conv_s2_rgb_input_299x299x3-i8": _conv((299, 299, 3), 32, 3, 2, relu6=True),
    "conv_s2_rgb_input_300x300x3-i8": _conv((300, 300, 3), 32, 3, 2, "same", relu6=True),
    "fully_connected_after_global_pool-i8": _pool_then_dense((7, 7, 64), 32),
    "fully_connected_per_tensor_batch16-i8": _dense(16, 96, 288),
    "fully_connected_per_tensor_bias-i8": _dense(4, 16, 8),
    "fully_connected_per_tensor_input_zp-i8": _dense(1, 96, 144),
    "dw_1x3_stride2_same_11x11x8": _depthwise((11, 11, 8), (1, 3), 2, "same"),
    "dw_3x5_stride2_same_11x11x8": _depthwise((11, 11, 8), (3, 5), 2, "same"),
    "dw_3x7_stride2_same_11x11x8": _depthwise((11, 11, 8), (3, 7), 2, "same"),
    "dw_5x3_stride2_same_11x11x8": _depthwise((11, 11, 8), (5, 3), 2, "same"),
    "dw_4x4_stride2_valid_12x12x8": _depthwise((12, 12, 8), 4, 2),
    "dw_4x6_stride2_valid_12x12x8": _depthwise((12, 12, 8), (4, 6), 2),
    "dw_6x6_stride2_valid_12x12x8": _depthwise((12, 12, 8), 6, 2),
    "dw_21x21_odd_frame_stride2": _depthwise((21, 21, 576), 3, 2, bias=False),
    "dw_4x4_stride2": _depthwise((27, 27, 128), 4, 2, "same"),
}
