"""Conv operator fixtures built with torq.testing.onnx_builder."""

from functools import partial

import numpy as np

from torq.testing.onnx_builder import BF16, F32, I8, I32, I64, OnnxGraph

KIND = "onnx"


def _dequantized(g, values, scale, zero_point=0):
    """DequantizeLinear of constant int8 weights or int32 bias (zero point of the same type)."""
    dtype = I32 if values.dtype == np.int32 else I8
    return g.dequantize(g.const(values, dtype), scale, zero_point, zp_dtype=dtype)


def _make_qdq_conv1d(
    *,
    quantized_weights_shape,
    channels=64,
    length=6401,
    output_zero_point_value=-7,
    output_scale_value=0.014526698738336563,
    weight_scale_value=0.000876599398907274,
    quantized_weights_low=-119,
    quantized_weights_high=128,
    quantized_bias_low=-7386,
    quantized_bias_high=6499,
    bias_scale_value=6.875288363517029e-06,
    **attrs
):
    """A QDQ conv1d, run as a 1xW conv2d between reshapes."""
    rng = np.random.default_rng(0)
    g = OnnxGraph(opset=19)
    x = g.input("x", F32, [1, channels, length])
    weights = _dequantized(
        g,
        rng.integers(quantized_weights_low, quantized_weights_high, quantized_weights_shape,
                     dtype=np.int8),
        weight_scale_value,
    )
    bias = _dequantized(
        g,
        rng.integers(quantized_bias_low, quantized_bias_high, [channels], dtype=np.int32),
        g.const([bias_scale_value], F32),
    )
    x = g.op("Reshape", x, g.const([1, channels, 1, length], I64))
    y = g.op("Conv", g.qdq(x, 0.007843135856091976, 0), weights, bias, **attrs)
    y = g.qdq(y, output_scale_value, output_zero_point_value)
    shape = [1, channels, length]
    return g.export(g.op("Reshape", y, g.const(shape, I64), dtype=F32, shape=shape))


def _make_conv(
    input_shape,
    weight_shape,
    output_shape,
    *,
    weight_range=None,
    weight_value=None,
    bias_range=None,
    bias_value=None,
    bias_first=False,
    activation=None,
    op="Conv",
    opset=22,
    **attrs
):
    rng = np.random.default_rng(0)
    g = OnnxGraph(opset)
    x = g.input("x", BF16, input_shape)
    channels = [output_shape[1]]
    # Preserve RNG order for cases whose original graph initialized bias first.
    bias = rng.uniform(*bias_range, channels) if bias_first else None
    if weight_range is not None:
        weights = rng.uniform(*weight_range, weight_shape)
    else:
        weights = np.full(weight_shape, weight_value)
    if bias is None and bias_range is not None:
        bias = rng.uniform(*bias_range, channels)
    elif bias is None and bias_value is not None:
        bias = np.full(channels, bias_value)
    inputs = [x, g.const(weights, BF16)] + ([] if bias is None else [g.const(bias, BF16)])
    output = {"dtype": BF16, "shape": output_shape}
    if activation is None:
        return g.export(g.op(op, *inputs, **attrs, **output))
    y = g.op(op, *inputs, **attrs)
    if activation == "relu":
        y = g.op("Relu", y, **output)
    elif activation == "clip":
        y = g.op("Clip", y, g.const(0.0, BF16), g.const(6.0, BF16), **output)
    elif activation == "clip_constants":  # Clip bounds from Constant nodes
        bounds = [g.op("Constant", value=g.const(v, BF16, name="value")) for v in (0.0, 6.0)]
        y = g.op("Clip", y, *bounds, **output)
    return g.export(y)


def make_conv_izp_i8():
    rng = np.random.default_rng(0)
    g = OnnxGraph(opset=19)
    x = g.input("x", F32, [1, 4, 1, 16])
    weights = _dequantized(g, rng.integers(1, 10, [4, 4, 1, 3], dtype=np.int8),
                           0.014546649530529976)
    bias = _dequantized(g, np.array([1, 3, 5, 7], dtype=np.int32), 0.00011364569945726544)
    y = g.op("Conv", g.qdq(x, 0.0078125, -23), weights, bias, dilations=[1, 1], group=1,
             kernel_shape=[1, 3], pads=[0, 1, 0, 1], strides=[1, 1])
    return g.export(g.qdq(y, 0.014546649530529976, 0, dtype=F32, shape=[1, 4, 1, 16]))


def _make_conv_relu_conv(
    *,
    convolved_shape,
    output_shape,
    weights1_shape,
    input_shape=(1, 8, 8, 40),
    consumer_group=1,
    consumer_pads=(1, 1, 1, 1),
    weights1_low=-0.38671875,
    weights1_high=0.439453125,
    bias1_low=-0.04541015625,
    bias1_high=0.203125,
    weights2_low=-0.3515625,
    weights2_high=0.33203125,
    bias2_low=-0.10107421875,
    bias2_high=0.0947265625,
    **attrs
):
    """Conv (attrs) -> Relu -> 3x3 stride-2 Conv (consumer_group, consumer_pads)."""
    rng = np.random.default_rng(0)
    g = OnnxGraph(opset=22)
    x = g.input("x", BF16, input_shape)
    weights1 = g.const(rng.uniform(weights1_low, weights1_high, weights1_shape), BF16)
    bias1 = g.const(rng.uniform(bias1_low, bias1_high, [8]), BF16)
    weights2 = g.const(
        rng.uniform(weights2_low, weights2_high, [8, 8 // consumer_group, 3, 3]), BF16
    )
    bias2 = g.const(rng.uniform(bias2_low, bias2_high, [8]), BF16)
    y = g.op("Conv", x, weights1, bias1, dtype=BF16, shape=convolved_shape, **attrs)
    y = g.op("Relu", y, dtype=BF16, shape=convolved_shape)
    y = g.op("Conv", y, weights2, bias2, auto_pad="NOTSET", dilations=[1, 1],
             group=consumer_group, kernel_shape=[3, 3], pads=list(consumer_pads), strides=[2, 2],
             dtype=BF16, shape=output_shape)
    return g.export(y)


def make_depthwise_conv1d_bf16_c64_w32():
    rng = np.random.default_rng(0)
    g = OnnxGraph(opset=22)
    shape = [1, 64, 32]
    x = g.input("x", BF16, shape)
    bias = g.const(rng.uniform(-1.0390625, 1.3046875, [64]), BF16)
    weights = g.const(rng.uniform(-2.671875, 2.359375, [64, 1, 3]), BF16)
    y = g.op("Conv", x, weights, bias, dilations=[1], group=64, kernel_shape=[3], pads=[1, 1],
             strides=[1], dtype=BF16, shape=shape)
    return g.export(g.op("Clip", y, g.const(0.0, BF16), g.const(6.0, BF16), dtype=BF16,
                         shape=shape))


def make_convtranspose1d_s2_crop_bf16():
    rng = np.random.default_rng(0)
    g = OnnxGraph(opset=22)
    x = g.input("x", BF16, [1, 1, 128])
    weights = g.const(rng.uniform(0.5234375, 2.5, [1, 1, 3]), BF16)
    y = g.op("ConvTranspose", x, weights, g.const([-6.75], BF16), dilations=[1], group=1,
             kernel_shape=[3], pads=[0, 0], strides=[2])
    crop = g.op("Slice", y, g.const([0, 0, 0], I64), g.const([1, 1, 256], I64), dtype=BF16,
                shape=[1, 1, 256])
    return g.export(crop)


def make_qdq_conv2d():
    rng = np.random.default_rng(0)
    g = OnnxGraph(opset=18)
    x = g.input("x", I8, [1, 3, 224, 224])
    weights = _dequantized(g, rng.integers(-127, 102, [64, 3, 7, 7], dtype=np.int8),
                           0.022440632805228233)
    bias = _dequantized(g, rng.integers(-30607, 25279, [64], dtype=np.int32),
                        g.const([0.00017600483261048794], F32))
    y = g.op("Conv", g.dequantize(x, 0.007843131199479103, -1), weights, bias,
             dilations=[1, 1], group=1, kernel_shape=[7, 7], pads=[3, 3, 3, 3], strides=[2, 2])
    return g.export(g.quantize(y, 0.13208869099617004, -22, dtype=I8, shape=[1, 64, 112, 112]))


def make_conv_qdq_sliced_remainder_i8():
    rng = np.random.default_rng(0)
    g = OnnxGraph(opset=19)
    x = g.input("x", F32, [1, 16, 12, 12])
    weights = _dequantized(g, rng.integers(-7, 8, [20, 16, 3, 3], dtype=np.int8),
                           0.014546649530529976)
    bias = _dequantized(g, rng.integers(-50, 48, [20], dtype=np.int32), 0.00011364569945726544)
    y = g.op("Conv", g.qdq(x, 0.0078125, -23), weights, bias, dilations=[1, 1], group=1,
             kernel_shape=[3, 3], pads=[1, 1, 1, 1], strides=[1, 1])
    return g.export(g.qdq(y, 0.0625, 0, dtype=F32, shape=[1, 20, 12, 12]))


# Conv attributes as the original models spelled them out.
_CONV_3X3_ATTRS = {"auto_pad": "NOTSET", "dilations": [1, 1], "kernel_shape": [3, 3],
               "pads": [1, 1, 1, 1]}

CASES = {
    "conv2d_bf16_8x8x3x3_s1_into_dw_s2_segment_output_8x7x40": partial(
        _make_conv_relu_conv, input_shape=[1, 8, 7, 40], convolved_shape=[1, 8, 7, 40],
        output_shape=[1, 8, 3, 20], weights1_shape=[8, 8, 3, 3], consumer_group=8,
        consumer_pads=[0, 0, 0, 1], **_CONV_3X3_ATTRS, group=1, strides=[1, 1],
    ),
    "conv2d_bf16_8x8x3x3_s1_into_dw_s2_valid_8x8x40": partial(
        _make_conv_relu_conv, input_shape=[1, 8, 8, 40], convolved_shape=[1, 8, 8, 40],
        output_shape=[1, 8, 3, 19], weights1_shape=[8, 8, 3, 3], consumer_group=8,
        consumer_pads=[0, 0, 0, 0], **_CONV_3X3_ATTRS, group=1, strides=[1, 1],
    ),
    "conv2d_bf16_8x8x3x3_s1_into_s2_segment_output_8x8x40": partial(
        _make_conv_relu_conv, convolved_shape=[1, 8, 8, 40], output_shape=[1, 8, 4, 20],
        weights1_shape=[8, 8, 3, 3], **_CONV_3X3_ATTRS, group=1, strides=[1, 1],
    ),
    "conv2d_bf16_8x8x3x3_s2_into_s2_segment_output_8x8x40": partial(
        _make_conv_relu_conv, convolved_shape=[1, 8, 4, 20], output_shape=[1, 8, 2, 10],
        weights1_shape=[8, 8, 3, 3], **_CONV_3X3_ATTRS, group=1, strides=[2, 2],
    ),
    "encoder_bf16_layer_Conv_2-sim-sl2610-v1": partial(
        _make_conv, [1, 1, 80000], [288, 1, 127], [1, 288, 1249], weight_range=(-1.6875, 1.71875),
        auto_pad="NOTSET", dilations=[1], group=1, kernel_shape=[127], pads=[0, 0], strides=[64],
    ),
    "conv1d_295_qdq_tiled": partial(
        _make_qdq_conv1d, quantized_weights_shape=[64, 64, 1, 11],
        dilations=[1, 1], group=1, kernel_shape=[1, 11], pads=[0, 5, 0, 5], strides=[1, 1],
    ),
    "conv1d_307_qdq_tiled": partial(
        _make_qdq_conv1d,
        output_zero_point_value=0,
        output_scale_value=0.014546649530529976,
        weight_scale_value=0.0009804072324186563,
        quantized_weights_low=-127,
        quantized_weights_high=113,
        quantized_weights_shape=[64, 64, 1, 7],
        quantized_bias_low=-5937,
        quantized_bias_high=8256,
        bias_scale_value=7.689467565796804e-06,
        dilations=[1, 1], group=1, kernel_shape=[1, 7], pads=[0, 3, 0, 3], strides=[1, 1],
    ),
    # Quantized conv1ds with a kernel wider than 7 columns, SAME padding: they run on the
    # EasyKernel 1D valid path. k9 is a multiple of the ALU kernel width (3); k11 is not.
    "conv1d-qdq-wide-kernel-k9-8x100-i8": partial(
        _make_qdq_conv1d, channels=8, length=100, quantized_weights_low=-128,
        quantized_weights_shape=[8, 8, 1, 9], quantized_bias_low=-1833, quantized_bias_high=1288,
        dilations=[1, 1], group=1, kernel_shape=[1, 9], pads=[0, 4, 0, 4], strides=[1, 1],
    ),
    "conv1d-qdq-wide-kernel-k11-8x100-i8": partial(
        _make_qdq_conv1d, channels=8, length=100, quantized_weights_low=-128,
        quantized_weights_shape=[8, 8, 1, 11], quantized_bias_low=-1393, quantized_bias_high=2979,
        dilations=[1, 1], group=1, kernel_shape=[1, 11], pads=[0, 5, 0, 5], strides=[1, 1],
    ),
    # LinalgSlicing splits its 20 output channels over the two SL2610 slices: 16 in the
    # scf.forall and a peeled remainder of 4, with a nonzero input zero point.
    "conv-qdq-sliced-remainder-i8": make_conv_qdq_sliced_remainder_i8,
    "conv1d_matmul_bf16_bias": partial(
        _make_conv, [1, 3, 4], [2, 3, 1], [1, 2, 4], weight_range=(1.0, 6.0),
        bias_range=(1.0, 2.0),
        auto_pad="NOTSET", dilations=[1], group=1, kernel_shape=[1], pads=[0, 0], strides=[1],
    ),
    "conv1d_matmul_bf16_tile_and_fuse": partial(
        _make_conv, [1, 288, 207], [1152, 288, 1], [1, 1152, 207], weight_value=0.010009765625,
        auto_pad="NOTSET", dilations=[1], group=1, kernel_shape=[1], pads=[0, 0], strides=[1],
    ),
    "conv1d_nobias_clip_bf16_16x16x3_s3_16x64": partial(
        _make_conv, [1, 16, 64], [16, 16, 3], [1, 16, 22], weight_range=(-0.14453125, 0.14453125),
        activation="clip_constants",
        dilations=[1], group=1, kernel_shape=[3], pads=[1, 1], strides=[3],
    ),
    "conv2d-nchw-clip-bf16": partial(
        _make_conv, [1, 3, 224, 224], [32, 3, 3, 3], [1, 32, 112, 112],
        weight_range=(-4.875, 5.6875), bias_range=(-5.59375, 6.4375), activation="clip",
        auto_pad="NOTSET", dilations=[1, 1], group=1, pads=[1, 1, 1, 1], strides=[2, 2],
    ),
    "conv2d_clip_bf16_32x3x56x56_k3x3_s1_g1": partial(
        _make_conv, [1, 3, 56, 56], [32, 3, 3, 3], [1, 32, 56, 56],
        weight_range=(-4.875, 5.6875), bias_range=(-5.59375, 6.4375), activation="clip",
        auto_pad="NOTSET", dilations=[1, 1], group=1, pads=[1, 1, 1, 1], strides=[1, 1],
    ),
    "conv2d_clip_bf16_32x3x56x56_k3x3_s2_g1": partial(
        _make_conv, [1, 3, 56, 56], [32, 3, 3, 3], [1, 32, 28, 28],
        weight_range=(-4.875, 5.6875), bias_range=(-5.59375, 6.4375), activation="clip",
        auto_pad="NOTSET", dilations=[1, 1], group=1, pads=[1, 1, 1, 1], strides=[2, 2],
    ),
    "conv2d_dw_clip_bf16_32x112x112_k3x3_g32": partial(
        _make_conv, [1, 32, 112, 112], [32, 1, 3, 3], [1, 32, 112, 112],
        weight_range=(-7.59375, 7.96875), bias_range=(-0.9296875, 5.90625), activation="clip",
        auto_pad="NOTSET", dilations=[1, 1], group=32, pads=[1, 1, 1, 1], strides=[1, 1],
    ),
    "conv2d_nobias_clip_bf16_8x8x3x3_s1_8x32x32": partial(
        _make_conv, [1, 8, 32, 32], [8, 8, 3, 3], [1, 8, 32, 32],
        weight_range=(-0.3828125, 0.298828125), activation="clip_constants",
        **_CONV_3X3_ATTRS, group=1, strides=[1, 1],
    ),
    "conv2d_nobias_relu_bf16_8x8x3x3_s1_8x32x32": partial(
        _make_conv, [1, 8, 32, 32], [8, 8, 3, 3], [1, 8, 32, 32],
        weight_range=(-0.3828125, 0.298828125), activation="relu",
        **_CONV_3X3_ATTRS, group=1, strides=[1, 1],
    ),
    "conv_izp_i8": make_conv_izp_i8,
    "depthwise-conv1d-bf16-c464-k5": partial(
        _make_conv, [1, 464, 320], [464, 1, 5], [1, 464, 320], weight_value=0.30078125,
        bias_value=0.10009765625, opset=19,
        dilations=[1], group=464, kernel_shape=[5], pads=[2, 2], strides=[1],
    ),
    "depthwise-conv1d-bf16-c64-w32": make_depthwise_conv1d_bf16_c64_w32,
    "depthwise_conv2d_bf16_8x3x3_s2_into_s2_segment_output_8x8x40": partial(
        _make_conv_relu_conv,
        convolved_shape=[1, 8, 4, 20],
        output_shape=[1, 8, 2, 10],
        weights1_shape=[8, 1, 3, 3],
        weights1_low=-0.84375,
        weights1_high=0.92578125,
        bias1_low=-0.142578125,
        bias1_high=0.1796875,
        weights2_low=-0.38671875,
        weights2_high=0.439453125,
        bias2_low=-0.1669921875,
        bias2_high=0.1962890625,
        **_CONV_3X3_ATTRS, group=8, strides=[2, 2],
    ),
    "depthwise_conv1d_bf16_stride2_valid_pad": partial(
        _make_conv, [1, 32, 128], [32, 1, 3], [1, 32, 64], weight_range=(-3.203125, 2.0),
        bias_range=(-0.75, 2.03125), bias_first=True,
        dilations=[1], group=32, kernel_shape=[3], pads=[0, 1], strides=[2],
    ),
    "dw1d_k5_d2_s1": partial(
        _make_conv, [1, 64, 1, 9], [64, 1, 1, 5], [1, 64, 1, 1],
        weight_range=(-0.447265625, 0.44140625), bias_range=(-0.4375, 0.43359375),
        dilations=[1, 2], group=64, kernel_shape=[1, 5], pads=[0, 0, 0, 0], strides=[1, 1],
    ),
    "dw1d_k5_d4_s1": partial(
        _make_conv, [1, 64, 1, 17], [64, 1, 1, 5], [1, 64, 1, 1],
        weight_range=(-0.44140625, 0.4453125), bias_range=(-0.419921875, 0.443359375),
        dilations=[1, 4], group=64, kernel_shape=[1, 5], pads=[0, 0, 0, 0], strides=[1, 1],
    ),
    "dw1d_k5_d8_s1": partial(
        _make_conv, [1, 64, 1, 33], [64, 1, 1, 5], [1, 64, 1, 1],
        weight_range=(-0.447265625, 0.44140625), bias_range=(-0.4375, 0.392578125),
        dilations=[1, 8], group=64, kernel_shape=[1, 5], pads=[0, 0, 0, 0], strides=[1, 1],
    ),
    "dw1d_k5_dilation16_stride1": partial(
        _make_conv, [1, 64, 1, 65], [64, 1, 1, 5], [1, 64, 1, 1],
        weight_range=(-0.4453125, 0.4453125), bias_range=(-0.4375, 0.44140625),
        dilations=[1, 16], group=64, kernel_shape=[1, 5], pads=[0, 0, 0, 0], strides=[1, 1],
    ),
    "dw_1x4_dilation_1x2_expanded_1x7": partial(
        _make_conv, [1, 64, 1, 10], [64, 1, 1, 4], [1, 64, 1, 4],
        weight_range=(-0.8984375, 0.8984375), bias_range=(-0.2001953125, 0.30078125),
        dilations=[1, 2], group=64, kernel_shape=[1, 4], pads=[0, 0, 0, 0], strides=[1, 1],
    ),
    "dw_bf16_f5x2_s2x1_2x256x2": partial(
        _make_conv, [1, 2, 256, 2], [2, 1, 5, 2], [1, 2, 128, 1],
        weight_range=(-0.095703125, 0.07080078125), bias_value=0.0,
        dilations=[1, 1], group=2, kernel_shape=[5, 2], pads=[2, 0, 2, 0], strides=[2, 1],
    ),
    "conv_transpose_bf16_stride_2x2_interleave": partial(
        _make_conv, [1, 8, 4, 16], [8, 8, 2, 2], [1, 8, 8, 32], weight_range=(-1.625, 1.125),
        op="ConvTranspose", kernel_shape=[2, 2], pads=[0, 0, 0, 0], strides=[2, 2],
    ),
    "conv_transpose_bf16_stride_5x2": partial(
        _make_conv, [1, 1, 16, 2], [1, 1, 5, 2], [1, 1, 32, 1],
        weight_range=(-0.062255859375, 0.07421875), bias_value=0.0, op="ConvTranspose",
        dilations=[1, 1], group=1, kernel_shape=[5, 2], output_padding=[1, 0],
        pads=[2, 1, 2, 1], strides=[2, 1],
    ),
    "convtranspose1d-s2-crop-bf16": make_convtranspose1d_s2_crop_bf16,
    "Conv2d_bf16_1x1x256x256": partial(
        _make_conv, [1, 256, 128, 1], [256, 256, 1, 1], [1, 256, 128, 1],
        bias_range=(-0.0625, 0.062255859375), weight_range=(-0.0625, 0.0625),
        dilations=[1, 1], group=1, kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1],
    ),
    "Conv2d_bf16_1x1x2x256": partial(
        _make_conv, [1, 2, 128, 1], [256, 2, 1, 1], [1, 256, 128, 1],
        weight_range=(-0.1455078125, 0.1552734375), bias_value=0.0,
        dilations=[1, 1], group=1, kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1],
    ),
    "Conv2d_bf16_1x1x64x8192": partial(
        _make_conv, [1, 64, 1, 1], [8192, 64, 1, 1], [1, 8192, 1, 1],
        bias_range=(-0.125, 0.125), weight_range=(-0.125, 0.125),
        dilations=[1, 1], group=1, kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1],
    ),
    "conv2d-relu": partial(
        _make_conv, [1, 192, 35, 35], [64, 192, 1, 1], [1, 64, 35, 35],
        bias_range=(-30.875, 11.625), weight_range=(-0.80859375, 2.46875), activation="relu",
        opset=18,
        dilations=[1, 1], group=1, kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1],
    ),
    "qdq-conv2d": make_qdq_conv2d,
    "ConvTranspose_bf16_1x1x512x512": partial(
        _make_conv, [1, 512, 16, 1], [512, 512, 1, 1], [1, 512, 16, 1],
        weight_range=(-0.2255859375, 0.2412109375), bias_value=0.0, op="ConvTranspose",
        dilations=[1, 1], group=1, kernel_shape=[1, 1], pads=[0, 0, 0, 0], strides=[1, 1],
    ),
    "conv2d-host": partial(
        _make_conv, [1, 3, 224, 224], [64, 3, 11, 11], [1, 64, 55, 55],
        bias_range=(-2.796875, 1.640625), bias_first=True, weight_range=(-0.78125, 0.93359375),
        opset=16,
        dilations=[1, 1], group=1, kernel_shape=[11, 11], pads=[2, 2, 2, 2], strides=[4, 4],
    ),
    "conv1d-bf16-k127-s64": partial(
        _make_conv, [1, 1, 191], [288, 1, 127], [1, 288, 2], weight_range=(-1.6796875, 1.71875),
        auto_pad="NOTSET", dilations=[1], group=1, kernel_shape=[127], pads=[0, 0], strides=[64],
    ),
    "conv1d-bf16-c288-k7-s3": partial(
        _make_conv, [1, 288, 7], [576, 288, 7], [1, 576, 1],
        bias_range=(-1.3515625, 0.494140625), weight_range=(-2.171875, 1.8671875),
        auto_pad="NOTSET", dilations=[1], group=1, kernel_shape=[7], pads=[0, 0], strides=[3],
    ),
    "conv1d-bf16-c576-k3-s2": partial(
        _make_conv, [1, 576, 5], [288, 576, 3], [1, 288, 2],
        bias_range=(-0.77734375, 3.390625), weight_range=(-3.8125, 1.203125),
        auto_pad="NOTSET", dilations=[1], group=1, kernel_shape=[3], pads=[0, 0], strides=[2],
    ),
}

CONFIG = {
    "Conv2d_bf16_1x1x64x8192": {"comparison": {"fp_max_tol": 1.1, "fp_avg_tol": 0.2}},
    # Gates of one int8 step, not two: a few outputs differ from the float reference by one step.
    "conv-qdq-sliced-remainder-i8": {
        "comparison": {"use_abs_tol_gate": 1.0, "fp_abs_tol_frac": 0.03},
    },
    "conv1d-qdq-wide-kernel-k9-8x100-i8": {
        "comparison": {"use_abs_tol_gate": 1.0, "fp_abs_tol_frac": 0.01},
    },
    "conv1d-qdq-wide-kernel-k11-8x100-i8": {
        "comparison": {"use_abs_tol_gate": 1.0, "fp_abs_tol_frac": 0.01},
    },
    # The reduced Moonshine encoder convs and conv2d-host keep the host/CSS fallback.
    "conv1d-bf16-c288-k7-s3": {"compiler_options": []},
    "conv1d-bf16-c576-k3-s2": {"compiler_options": []},
    "conv1d-bf16-k127-s64": {
        "compiler_options": [],
        "comparison": {"fp_avg_tol": 0.2, "fp_max_tol": 1.0},
    },
    "conv2d-host": {"compiler_options": [], "comparison": {"epsilon": 1e-3}},
    # Tiled int8 QDQ conv W-shift regressions, one per conv lowering path: conv1d_307 (kw=7)
    # goes through the EK kernel; conv1d_295 (kw=11) takes the legacy path, whose baseOffset
    # W-correction the fix restores (without it, ~80% of the outputs are wrong). The residual
    # is bf16 requant rounding. conv1d_295's 1x64x6401 output spans a wide dynamic range
    # (~1.0 down to ~1e-4), so the near-zero tail trips the relative-error metric on a few
    # percent of the elements, more on sr200 (~2%) than on sl2610.
    "conv1d_295_qdq_tiled": {"comparison": {"fp_max_tol": 0.2, "allowed_wrong": 0.05}},
    "conv1d_307_qdq_tiled": {"comparison": {"fp_max_tol": 0.2, "allowed_wrong": 0.02}},
    "conv1d_matmul_bf16_tile_and_fuse": {"comparison": {"fp_max_tol": 0.05}},
    # The nonzero input zero point (si8 -23) must fold into the bias (the NSS does not subtract
    # it). The residual is bf16 requant rounding.
    "conv_izp_i8": {"comparison": {"fp_max_tol": 0.2, "allowed_wrong": 0.06}},
    # The S2D fold accumulates this stride-64 conv1d (audio-encoder layer) in a different order
    # than the llvm-cpu f32 reference: a few near-cancellation outputs (~3.5e-5) are ~1 bf16 ULP
    # off, beyond the default relative fp_max_tol. Measured on sim-sl2610: 9/359712 differ (max
    # relative 0.016). A real regression would affect thousands of elements.
    "encoder_bf16_layer_Conv_2-sim-sl2610-v1": {"comparison": {"allowed_wrong": 1e-4}},
}
