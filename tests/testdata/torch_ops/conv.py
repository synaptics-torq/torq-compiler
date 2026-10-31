"""Conv operator fixtures built with ONNX GraphSurgeon."""

from functools import partial

import ml_dtypes
import numpy as np
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def _make_qdq_conv1d(
    *,
    channels=64,
    length=6401,
    output_zero_point_value=-7,
    output_scale_value=0.014526698738336563,
    weight_scale_value=0.000876599398907274,
    quantized_weights_low=-119,
    quantized_weights_high=128,
    quantized_weights_shape,
    quantized_bias_low=-7386,
    quantized_bias_high=6499,
    bias_scale_value=6.875288363517029e-06,
    attrs
) -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    x = gs.Variable("x", dtype=np.float32, shape=[1, channels, length])
    output = gs.Variable("output", dtype=np.float32, shape=[1, channels, length])
    nchw_shape = gs.Constant("nchw_shape", np.array([1, channels, 1, length], dtype=np.int64))
    output_shape = gs.Constant("output_shape", np.array([1, channels, length], dtype=np.int64))
    input_zero_point = gs.Constant("input_zero_point", np.full([], 0, dtype=np.int8))
    input_scale = gs.Constant("input_scale", np.full([], 0.007843135856091976, dtype=np.float32))
    output_zero_point = gs.Constant(
        "output_zero_point", np.full([], output_zero_point_value, dtype=np.int8)
    )
    output_scale = gs.Constant("output_scale", np.full([], output_scale_value, dtype=np.float32))
    weight_zero_point = gs.Constant("weight_zero_point", np.full([], 0, dtype=np.int8))
    weight_scale = gs.Constant("weight_scale", np.full([], weight_scale_value, dtype=np.float32))
    quantized_weights = gs.Constant(
        "quantized_weights",
        rng.integers(
            quantized_weights_low, quantized_weights_high, quantized_weights_shape, dtype=np.int8
        ),
    )
    quantized_bias = gs.Constant(
        "quantized_bias",
        rng.integers(quantized_bias_low, quantized_bias_high, [channels], dtype=np.int32),
    )
    bias_scale = gs.Constant("bias_scale", np.full([1], bias_scale_value, dtype=np.float32))
    bias_zero_point = gs.Constant("bias_zero_point", np.full([], 0, dtype=np.int32))
    bias = gs.Variable("bias")
    weights = gs.Variable("weights")
    reshaped_input = gs.Variable("reshaped_input")
    quantized_input = gs.Variable("quantized_input")
    dequantized_input = gs.Variable("dequantized_input")
    convolved = gs.Variable("convolved")
    quantized_output = gs.Variable("quantized_output")
    dequantized_output = gs.Variable("dequantized_output")
    graph = gs.Graph(
        nodes=[
            gs.Node(
                "DequantizeLinear",
                inputs=[quantized_bias, bias_scale, bias_zero_point],
                outputs=[bias],
            ),
            gs.Node(
                "DequantizeLinear",
                inputs=[quantized_weights, weight_scale, weight_zero_point],
                outputs=[weights],
            ),
            gs.Node("Reshape", inputs=[x, nchw_shape], outputs=[reshaped_input]),
            gs.Node(
                "QuantizeLinear",
                inputs=[reshaped_input, input_scale, input_zero_point],
                outputs=[quantized_input],
            ),
            gs.Node(
                "DequantizeLinear",
                inputs=[quantized_input, input_scale, input_zero_point],
                outputs=[dequantized_input],
            ),
            gs.Node(
                "Conv", inputs=[dequantized_input, weights, bias], outputs=[convolved], attrs=attrs
            ),
            gs.Node(
                "QuantizeLinear",
                inputs=[convolved, output_scale, output_zero_point],
                outputs=[quantized_output],
            ),
            gs.Node(
                "DequantizeLinear",
                inputs=[quantized_output, output_scale, output_zero_point],
                outputs=[dequantized_output],
            ),
            gs.Node("Reshape", inputs=[dequantized_output, output_shape], outputs=[output]),
        ],
        inputs=[x],
        outputs=[output],
        opset=19,
    )
    return gs.export_onnx(graph, ir_version=10)


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
) -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    x = gs.Variable("x", dtype=ml_dtypes.bfloat16, shape=input_shape)
    output = gs.Variable("output", dtype=ml_dtypes.bfloat16, shape=output_shape)
    # Preserve RNG order for cases whose original graph initialized bias first.
    bias = None
    if bias_first:
        bias = gs.Constant(
            "bias", rng.uniform(*bias_range, [output_shape[1]]).astype(ml_dtypes.bfloat16)
        )
    if weight_range is not None:
        values = rng.uniform(*weight_range, weight_shape).astype(ml_dtypes.bfloat16)
    else:
        values = np.full(weight_shape, weight_value, dtype=ml_dtypes.bfloat16)
    weights = gs.Constant("weights", values)
    if bias_range is not None and bias is None:
        bias = gs.Constant(
            "bias", rng.uniform(*bias_range, [output_shape[1]]).astype(ml_dtypes.bfloat16)
        )
    elif bias_value is not None:
        bias = gs.Constant("bias", np.full([output_shape[1]], bias_value, dtype=ml_dtypes.bfloat16))
    inputs = [x, weights] if bias is None else [x, weights, bias]
    conv = gs.Variable("conv") if activation else output
    nodes = [gs.Node(op, inputs=inputs, outputs=[conv], attrs=attrs)]
    if activation == "relu":
        nodes.append(gs.Node("Relu", inputs=[conv], outputs=[output]))
    elif activation == "clip_constants":
        minimum, maximum = gs.Variable("constant"), gs.Variable("constant_2")
        for tensor, value in ((minimum, 0.0), (maximum, 6.0)):
            nodes.append(
                gs.Node(
                    "Constant",
                    outputs=[tensor],
                    attrs={
                        "value": gs.Constant("value", np.full([], value, dtype=ml_dtypes.bfloat16))
                    },
                )
            )
        nodes.append(gs.Node("Clip", inputs=[conv, minimum, maximum], outputs=[output]))
    elif activation == "clip":
        minimum = gs.Constant("min_value", np.full([], 0.0, dtype=ml_dtypes.bfloat16))
        maximum = gs.Constant("max_value", np.full([], 6.0, dtype=ml_dtypes.bfloat16))
        nodes.append(gs.Node("Clip", inputs=[conv, minimum, maximum], outputs=[output]))
    graph = gs.Graph(nodes=nodes, inputs=[x], outputs=[output], opset=opset)
    return gs.export_onnx(graph, ir_version=10)


def make_conv_izp_i8() -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    x = gs.Variable("x", dtype=np.float32, shape=[1, 4, 1, 16])
    output = gs.Variable("output", dtype=np.float32, shape=[1, 4, 1, 16])
    scale = gs.Constant("scale", np.full([], 0.0078125, dtype=np.float32))
    zero_point = gs.Constant("zero_point", np.full([], -23, dtype=np.int8))
    scale_2 = gs.Constant("scale_2", np.full([], 0.014546649530529976, dtype=np.float32))
    zero_point_2 = gs.Constant("zero_point_2", np.full([], 0, dtype=np.int8))
    quantized = gs.Constant("quantized", rng.integers(1, 10, [4, 4, 1, 3], dtype=np.int8))
    scale_3 = gs.Constant("scale_3", np.full([], 0.00011364569945726544, dtype=np.float32))
    zero_point_3 = gs.Constant("zero_point_3", np.full([], 0, dtype=np.int32))
    quantized_2 = gs.Constant("quantized_2", np.array([1, 3, 5, 7], dtype=np.int32))
    scale_4 = gs.Constant("scale_4", np.full([], 0.014546649530529976, dtype=np.float32))
    zero_point_4 = gs.Constant("zero_point_4", np.full([], 0, dtype=np.int8))
    dequantizelinear = gs.Variable("dequantizelinear")
    dequantizelinear_2 = gs.Variable("dequantizelinear_2")
    quantizelinear = gs.Variable("quantizelinear")
    dequantizelinear_3 = gs.Variable("dequantizelinear_3")
    conv = gs.Variable("conv")
    quantizelinear_2 = gs.Variable("quantizelinear_2")
    graph = gs.Graph(
        nodes=[
            gs.Node(
                "DequantizeLinear",
                inputs=[quantized_2, scale_3, zero_point_3],
                outputs=[dequantizelinear],
            ),
            gs.Node(
                "DequantizeLinear",
                inputs=[quantized, scale_2, zero_point_2],
                outputs=[dequantizelinear_2],
            ),
            gs.Node("QuantizeLinear", inputs=[x, scale, zero_point], outputs=[quantizelinear]),
            gs.Node(
                "DequantizeLinear",
                inputs=[quantizelinear, scale, zero_point],
                outputs=[dequantizelinear_3],
            ),
            gs.Node(
                "Conv",
                inputs=[dequantizelinear_3, dequantizelinear_2, dequantizelinear],
                outputs=[conv],
                attrs={
                    "dilations": [1, 1],
                    "group": 1,
                    "kernel_shape": [1, 3],
                    "pads": [0, 1, 0, 1],
                    "strides": [1, 1],
                },
            ),
            gs.Node(
                "QuantizeLinear", inputs=[conv, scale_4, zero_point_4], outputs=[quantizelinear_2]
            ),
            gs.Node(
                "DequantizeLinear",
                inputs=[quantizelinear_2, scale_4, zero_point_4],
                outputs=[output],
            ),
        ],
        inputs=[x],
        outputs=[output],
        opset=19,
    )
    return gs.export_onnx(graph, ir_version=10)


def _make_conv_relu_conv(
    *,
    convolved_shape,
    output_shape,
    input_shape=(1, 8, 8, 40),
    consumer_group=1,
    consumer_pads=(1, 1, 1, 1),
    weights1_low=-0.38671875,
    weights1_high=0.439453125,
    weights1_shape,
    bias1_low=-0.04541015625,
    bias1_high=0.203125,
    weights2_low=-0.3515625,
    weights2_high=0.33203125,
    bias2_low=-0.10107421875,
    bias2_high=0.0947265625,
    attrs
) -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    x = gs.Variable("x", dtype=ml_dtypes.bfloat16, shape=input_shape)
    convolved = gs.Variable("convolved", dtype=ml_dtypes.bfloat16, shape=convolved_shape)
    activated = gs.Variable("activated", dtype=ml_dtypes.bfloat16, shape=convolved_shape)
    output = gs.Variable("output", dtype=ml_dtypes.bfloat16, shape=output_shape)
    weights1 = gs.Constant(
        "weights1",
        rng.uniform(weights1_low, weights1_high, weights1_shape).astype(ml_dtypes.bfloat16),
    )
    bias1 = gs.Constant("bias1", rng.uniform(bias1_low, bias1_high, [8]).astype(ml_dtypes.bfloat16))
    weights2 = gs.Constant(
        "weights2",
        rng.uniform(weights2_low, weights2_high, [8, 8 // consumer_group, 3, 3]).astype(ml_dtypes.bfloat16),
    )
    bias2 = gs.Constant("bias2", rng.uniform(bias2_low, bias2_high, [8]).astype(ml_dtypes.bfloat16))
    graph = gs.Graph(
        nodes=[
            gs.Node("Conv", inputs=[x, weights1, bias1], outputs=[convolved], attrs=attrs),
            gs.Node("Relu", inputs=[convolved], outputs=[activated]),
            gs.Node(
                "Conv",
                inputs=[activated, weights2, bias2],
                outputs=[output],
                attrs={
                    "auto_pad": "NOTSET",
                    "dilations": [1, 1],
                    "group": consumer_group,
                    "kernel_shape": [3, 3],
                    "pads": list(consumer_pads),
                    "strides": [2, 2],
                },
            ),
        ],
        inputs=[x],
        outputs=[output],
        opset=22,
    )
    return gs.export_onnx(graph, ir_version=10)


def make_depthwise_conv1d_bf16_c64_w32() -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    x = gs.Variable("x", dtype=ml_dtypes.bfloat16, shape=[1, 64, 32])
    convolved = gs.Variable("convolved", dtype=ml_dtypes.bfloat16, shape=[1, 64, 32])
    output = gs.Variable("output", dtype=ml_dtypes.bfloat16, shape=[1, 64, 32])
    maximum = gs.Constant("maximum", np.full([], 6.0, dtype=ml_dtypes.bfloat16))
    minimum = gs.Constant("minimum", np.full([], 0.0, dtype=ml_dtypes.bfloat16))
    bias = gs.Constant("bias", rng.uniform(-1.0390625, 1.3046875, [64]).astype(ml_dtypes.bfloat16))
    weights = gs.Constant(
        "weights", rng.uniform(-2.671875, 2.359375, [64, 1, 3]).astype(ml_dtypes.bfloat16)
    )
    graph = gs.Graph(
        nodes=[
            gs.Node(
                "Conv",
                inputs=[x, weights, bias],
                outputs=[convolved],
                attrs={
                    "dilations": [1],
                    "group": 64,
                    "kernel_shape": [3],
                    "pads": [1, 1],
                    "strides": [1],
                },
            ),
            gs.Node("Clip", inputs=[convolved, minimum, maximum], outputs=[output]),
        ],
        inputs=[x],
        outputs=[output],
        opset=22,
    )
    return gs.export_onnx(graph, ir_version=10)


def make_convtranspose1d_s2_crop_bf16() -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    x = gs.Variable("x", dtype=ml_dtypes.bfloat16, shape=[1, 1, 128])
    output = gs.Variable("output", dtype=ml_dtypes.bfloat16, shape=[1, 1, 256])
    weights = gs.Constant(
        "weights", rng.uniform(0.5234375, 2.5, [1, 1, 3]).astype(ml_dtypes.bfloat16)
    )
    bias = gs.Constant("bias", np.full([1], -6.75, dtype=ml_dtypes.bfloat16))
    starts = gs.Constant("starts", np.full([3], 0, dtype=np.int64))
    ends = gs.Constant("ends", np.array([1, 1, 256], dtype=np.int64))
    convtranspose = gs.Variable("convtranspose")
    graph = gs.Graph(
        nodes=[
            gs.Node(
                "ConvTranspose",
                inputs=[x, weights, bias],
                outputs=[convtranspose],
                attrs={
                    "dilations": [1],
                    "group": 1,
                    "kernel_shape": [3],
                    "pads": [0, 0],
                    "strides": [2],
                },
            ),
            gs.Node("Slice", inputs=[convtranspose, starts, ends], outputs=[output]),
        ],
        inputs=[x],
        outputs=[output],
        opset=22,
    )
    return gs.export_onnx(graph, ir_version=10)


def make_qdq_conv2d() -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    x = gs.Variable('x', dtype=np.int8, shape=[1, 3, 224, 224])
    output = gs.Variable('output', dtype=np.int8, shape=[1, 64, 112, 112])
    zero_point = gs.Constant('zero_point', np.full([], -1, dtype=np.int8))
    scale = gs.Constant('scale', np.full([], 0.007843131199479103, dtype=np.float32))
    zero_point_2 = gs.Constant('zero_point_2', np.full([], -22, dtype=np.int8))
    scale_2 = gs.Constant('scale_2', np.full([], 0.13208869099617004, dtype=np.float32))
    zero_point_3 = gs.Constant('zero_point_3', np.full([], 0, dtype=np.int8))
    scale_3 = gs.Constant('scale_3', np.full([], 0.022440632805228233, dtype=np.float32))
    quantized = gs.Constant('quantized', rng.integers(-127, 102, [64, 3, 7, 7], dtype=np.int8))
    quantized_2 = gs.Constant('quantized_2', rng.integers(-30607, 25279, [64], dtype=np.int32))
    scale_4 = gs.Constant('scale_4', np.full([1], 0.00017600483261048794, dtype=np.float32))
    zero_point_4 = gs.Constant('zero_point_4', np.full([], 0, dtype=np.int32))
    dequantizelinear = gs.Variable('dequantizelinear')
    dequantizelinear_2 = gs.Variable('dequantizelinear_2')
    dequantizelinear_3 = gs.Variable('dequantizelinear_3')
    conv = gs.Variable('conv')
    graph = gs.Graph(
        nodes=[
            gs.Node('DequantizeLinear', inputs=[quantized, scale_3, zero_point_3], outputs=[dequantizelinear]),
            gs.Node('DequantizeLinear', inputs=[quantized_2, scale_4, zero_point_4], outputs=[dequantizelinear_2]),
            gs.Node('DequantizeLinear', inputs=[x, scale, zero_point], outputs=[dequantizelinear_3]),
            gs.Node('Conv', inputs=[dequantizelinear_3, dequantizelinear, dequantizelinear_2], outputs=[conv], attrs={'dilations': [1, 1], 'group': 1, 'kernel_shape': [7, 7], 'pads': [3, 3, 3, 3], 'strides': [2, 2]}),
            gs.Node('QuantizeLinear', inputs=[conv, scale_2, zero_point_2], outputs=[output]),
        ],
        inputs=[x],
        outputs=[output],
        opset=18,
    )
    return gs.export_onnx(graph, ir_version=10)


def make_conv_qdq_sliced_remainder_i8() -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    x = gs.Variable("x", dtype=np.float32, shape=[1, 16, 12, 12])
    output = gs.Variable("output", dtype=np.float32, shape=[1, 20, 12, 12])
    input_scale = gs.Constant("input_scale", np.full([], 0.0078125, dtype=np.float32))
    input_zero_point = gs.Constant("input_zero_point", np.full([], -23, dtype=np.int8))
    weight_scale = gs.Constant("weight_scale", np.full([], 0.014546649530529976, dtype=np.float32))
    weight_zero_point = gs.Constant("weight_zero_point", np.full([], 0, dtype=np.int8))
    quantized_weights = gs.Constant(
        "quantized_weights", rng.integers(-7, 8, [20, 16, 3, 3], dtype=np.int8)
    )
    bias_scale = gs.Constant("bias_scale", np.full([], 0.00011364569945726544, dtype=np.float32))
    bias_zero_point = gs.Constant("bias_zero_point", np.full([], 0, dtype=np.int32))
    quantized_bias = gs.Constant("quantized_bias", rng.integers(-50, 48, [20], dtype=np.int32))
    output_scale = gs.Constant("output_scale", np.full([], 0.0625, dtype=np.float32))
    output_zero_point = gs.Constant("output_zero_point", np.full([], 0, dtype=np.int8))
    bias = gs.Variable("bias")
    weights = gs.Variable("weights")
    quantized_input = gs.Variable("quantized_input")
    dequantized_input = gs.Variable("dequantized_input")
    convolved = gs.Variable("convolved")
    quantized_output = gs.Variable("quantized_output")
    graph = gs.Graph(
        nodes=[
            gs.Node(
                "DequantizeLinear",
                inputs=[quantized_bias, bias_scale, bias_zero_point],
                outputs=[bias],
            ),
            gs.Node(
                "DequantizeLinear",
                inputs=[quantized_weights, weight_scale, weight_zero_point],
                outputs=[weights],
            ),
            gs.Node(
                "QuantizeLinear",
                inputs=[x, input_scale, input_zero_point],
                outputs=[quantized_input],
            ),
            gs.Node(
                "DequantizeLinear",
                inputs=[quantized_input, input_scale, input_zero_point],
                outputs=[dequantized_input],
            ),
            gs.Node(
                "Conv",
                inputs=[dequantized_input, weights, bias],
                outputs=[convolved],
                attrs={
                    "dilations": [1, 1],
                    "group": 1,
                    "kernel_shape": [3, 3],
                    "pads": [1, 1, 1, 1],
                    "strides": [1, 1],
                },
            ),
            gs.Node(
                "QuantizeLinear",
                inputs=[convolved, output_scale, output_zero_point],
                outputs=[quantized_output],
            ),
            gs.Node(
                "DequantizeLinear",
                inputs=[quantized_output, output_scale, output_zero_point],
                outputs=[output],
            ),
        ],
        inputs=[x],
        outputs=[output],
        opset=19,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "conv2d_bf16_8x8x3x3_s1_into_dw_s2_segment_output_8x7x40": partial(
        _make_conv_relu_conv,
        input_shape=[1, 8, 7, 40],
        convolved_shape=[1, 8, 7, 40],
        output_shape=[1, 8, 3, 20],
        weights1_shape=[8, 8, 3, 3],
        consumer_group=8,
        consumer_pads=[0, 0, 0, 1],
        attrs={
            "auto_pad": "NOTSET",
            "dilations": [1, 1],
            "group": 1,
            "kernel_shape": [3, 3],
            "pads": [1, 1, 1, 1],
            "strides": [1, 1],
        },
    ),
    "conv2d_bf16_8x8x3x3_s1_into_dw_s2_valid_8x8x40": partial(
        _make_conv_relu_conv,
        input_shape=[1, 8, 8, 40],
        convolved_shape=[1, 8, 8, 40],
        output_shape=[1, 8, 3, 19],
        weights1_shape=[8, 8, 3, 3],
        consumer_group=8,
        consumer_pads=[0, 0, 0, 0],
        attrs={
            "auto_pad": "NOTSET",
            "dilations": [1, 1],
            "group": 1,
            "kernel_shape": [3, 3],
            "pads": [1, 1, 1, 1],
            "strides": [1, 1],
        },
    ),
    "conv2d_bf16_8x8x3x3_s1_into_s2_segment_output_8x8x40": partial(
        _make_conv_relu_conv,
        convolved_shape=[1, 8, 8, 40],
        output_shape=[1, 8, 4, 20],
        weights1_shape=[8, 8, 3, 3],
        attrs={
            "auto_pad": "NOTSET",
            "dilations": [1, 1],
            "group": 1,
            "kernel_shape": [3, 3],
            "pads": [1, 1, 1, 1],
            "strides": [1, 1],
        },
    ),
    "conv2d_bf16_8x8x3x3_s2_into_s2_segment_output_8x8x40": partial(
        _make_conv_relu_conv,
        attrs={
            "auto_pad": "NOTSET",
            "dilations": [1, 1],
            "group": 1,
            "kernel_shape": [3, 3],
            "pads": [1, 1, 1, 1],
            "strides": [2, 2],
        },
        convolved_shape=[1, 8, 4, 20],
        output_shape=[1, 8, 2, 10],
        weights1_shape=[8, 8, 3, 3],
    ),
    "encoder_bf16_layer_Conv_2-sim-sl2610-v1": partial(
        _make_conv,
        [1, 1, 80000],
        [288, 1, 127],
        [1, 288, 1249],
        weight_range=(-1.6875, 1.71875),
        auto_pad="NOTSET",
        dilations=[1],
        group=1,
        kernel_shape=[127],
        pads=[0, 0],
        strides=[64],
    ),
    "conv1d_295_qdq_tiled": partial(
        _make_qdq_conv1d,
        quantized_weights_shape=[64, 64, 1, 11],
        attrs={
            "dilations": [1, 1],
            "group": 1,
            "kernel_shape": [1, 11],
            "pads": [0, 5, 0, 5],
            "strides": [1, 1],
        },
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
        attrs={
            "dilations": [1, 1],
            "group": 1,
            "kernel_shape": [1, 7],
            "pads": [0, 3, 0, 3],
            "strides": [1, 1],
        },
    ),
    # Quantized conv1ds with a kernel wider than 7 columns, SAME padding: they run on the
    # EasyKernel 1D valid path. k9 is a multiple of the ALU kernel width (3); k11 is not.
    "conv1d-qdq-wide-kernel-k9-8x100-i8": partial(
        _make_qdq_conv1d,
        channels=8,
        length=100,
        quantized_weights_low=-128,
        quantized_weights_shape=[8, 8, 1, 9],
        quantized_bias_low=-1833,
        quantized_bias_high=1288,
        attrs={
            "dilations": [1, 1],
            "group": 1,
            "kernel_shape": [1, 9],
            "pads": [0, 4, 0, 4],
            "strides": [1, 1],
        },
    ),
    "conv1d-qdq-wide-kernel-k11-8x100-i8": partial(
        _make_qdq_conv1d,
        channels=8,
        length=100,
        quantized_weights_low=-128,
        quantized_weights_shape=[8, 8, 1, 11],
        quantized_bias_low=-1393,
        quantized_bias_high=2979,
        attrs={
            "dilations": [1, 1],
            "group": 1,
            "kernel_shape": [1, 11],
            "pads": [0, 5, 0, 5],
            "strides": [1, 1],
        },
    ),
    # LinalgSlicing splits its 20 output channels over the two SL2610 slices: 16 in the
    # scf.forall and a peeled remainder of 4, with a nonzero input zero point.
    "conv-qdq-sliced-remainder-i8": make_conv_qdq_sliced_remainder_i8,
    "conv1d_matmul_bf16_bias": partial(
        _make_conv,
        [1, 3, 4],
        [2, 3, 1],
        [1, 2, 4],
        weight_range=(1.0, 6.0),
        bias_range=(1.0, 2.0),
        auto_pad="NOTSET",
        dilations=[1],
        group=1,
        kernel_shape=[1],
        pads=[0, 0],
        strides=[1],
    ),
    "conv1d_matmul_bf16_tile_and_fuse": partial(
        _make_conv,
        [1, 288, 207],
        [1152, 288, 1],
        [1, 1152, 207],
        weight_value=0.010009765625,
        auto_pad="NOTSET",
        dilations=[1],
        group=1,
        kernel_shape=[1],
        pads=[0, 0],
        strides=[1],
    ),
    "conv1d_nobias_clip_bf16_16x16x3_s3_16x64": partial(
        _make_conv,
        [1, 16, 64],
        [16, 16, 3],
        [1, 16, 22],
        weight_range=(-0.14453125, 0.14453125),
        activation="clip_constants",
        dilations=[1],
        group=1,
        kernel_shape=[3],
        pads=[1, 1],
        strides=[3],
    ),
    "conv2d-nchw-clip-bf16": partial(
        _make_conv,
        [1, 3, 224, 224],
        [32, 3, 3, 3],
        [1, 32, 112, 112],
        weight_range=(-4.875, 5.6875),
        bias_range=(-5.59375, 6.4375),
        activation="clip",
        auto_pad="NOTSET",
        dilations=[1, 1],
        group=1,
        pads=[1, 1, 1, 1],
        strides=[2, 2],
    ),
    "conv2d_clip_bf16_32x3x56x56_k3x3_s1_g1": partial(
        _make_conv,
        [1, 3, 56, 56],
        [32, 3, 3, 3],
        [1, 32, 56, 56],
        weight_range=(-4.875, 5.6875),
        bias_range=(-5.59375, 6.4375),
        activation="clip",
        auto_pad="NOTSET",
        dilations=[1, 1],
        group=1,
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    ),
    "conv2d_clip_bf16_32x3x56x56_k3x3_s2_g1": partial(
        _make_conv,
        [1, 3, 56, 56],
        [32, 3, 3, 3],
        [1, 32, 28, 28],
        weight_range=(-4.875, 5.6875),
        bias_range=(-5.59375, 6.4375),
        activation="clip",
        auto_pad="NOTSET",
        dilations=[1, 1],
        group=1,
        pads=[1, 1, 1, 1],
        strides=[2, 2],
    ),
    "conv2d_dw_clip_bf16_32x112x112_k3x3_g32": partial(
        _make_conv,
        [1, 32, 112, 112],
        [32, 1, 3, 3],
        [1, 32, 112, 112],
        weight_range=(-7.59375, 7.96875),
        bias_range=(-0.9296875, 5.90625),
        activation="clip",
        auto_pad="NOTSET",
        dilations=[1, 1],
        group=32,
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    ),
    "conv2d_nobias_clip_bf16_8x8x3x3_s1_8x32x32": partial(
        _make_conv,
        [1, 8, 32, 32],
        [8, 8, 3, 3],
        [1, 8, 32, 32],
        weight_range=(-0.3828125, 0.298828125),
        activation="clip_constants",
        auto_pad="NOTSET",
        dilations=[1, 1],
        group=1,
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    ),
    "conv2d_nobias_relu_bf16_8x8x3x3_s1_8x32x32": partial(
        _make_conv,
        [1, 8, 32, 32],
        [8, 8, 3, 3],
        [1, 8, 32, 32],
        weight_range=(-0.3828125, 0.298828125),
        activation="relu",
        auto_pad="NOTSET",
        dilations=[1, 1],
        group=1,
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        strides=[1, 1],
    ),
    "conv_izp_i8": make_conv_izp_i8,
    "depthwise-conv1d-bf16-c464-k5": partial(
        _make_conv,
        [1, 464, 320],
        [464, 1, 5],
        [1, 464, 320],
        weight_value=0.30078125,
        bias_value=0.10009765625,
        opset=19,
        dilations=[1],
        group=464,
        kernel_shape=[5],
        pads=[2, 2],
        strides=[1],
    ),
    "depthwise-conv1d-bf16-c64-w32": make_depthwise_conv1d_bf16_c64_w32,
    "depthwise_conv2d_bf16_8x3x3_s2_into_s2_segment_output_8x8x40": partial(
        _make_conv_relu_conv,
        weights1_low=-0.84375,
        weights1_high=0.92578125,
        weights1_shape=[8, 1, 3, 3],
        bias1_low=-0.142578125,
        bias1_high=0.1796875,
        weights2_low=-0.38671875,
        weights2_high=0.439453125,
        bias2_low=-0.1669921875,
        bias2_high=0.1962890625,
        attrs={
            "auto_pad": "NOTSET",
            "dilations": [1, 1],
            "group": 8,
            "kernel_shape": [3, 3],
            "pads": [1, 1, 1, 1],
            "strides": [2, 2],
        },
        convolved_shape=[1, 8, 4, 20],
        output_shape=[1, 8, 2, 10],
    ),
    "depthwise_conv1d_bf16_stride2_valid_pad": partial(
        _make_conv,
        [1, 32, 128],
        [32, 1, 3],
        [1, 32, 64],
        weight_range=(-3.203125, 2.0),
        bias_range=(-0.75, 2.03125),
        bias_first=True,
        dilations=[1],
        group=32,
        kernel_shape=[3],
        pads=[0, 1],
        strides=[2],
    ),
    "dw1d_k5_d2_s1": partial(
        _make_conv,
        [1, 64, 1, 9],
        [64, 1, 1, 5],
        [1, 64, 1, 1],
        weight_range=(-0.447265625, 0.44140625),
        bias_range=(-0.4375, 0.43359375),
        dilations=[1, 2],
        group=64,
        kernel_shape=[1, 5],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    ),
    "dw1d_k5_d4_s1": partial(
        _make_conv,
        [1, 64, 1, 17],
        [64, 1, 1, 5],
        [1, 64, 1, 1],
        weight_range=(-0.44140625, 0.4453125),
        bias_range=(-0.419921875, 0.443359375),
        dilations=[1, 4],
        group=64,
        kernel_shape=[1, 5],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    ),
    "dw1d_k5_d8_s1": partial(
        _make_conv,
        [1, 64, 1, 33],
        [64, 1, 1, 5],
        [1, 64, 1, 1],
        weight_range=(-0.447265625, 0.44140625),
        bias_range=(-0.4375, 0.392578125),
        dilations=[1, 8],
        group=64,
        kernel_shape=[1, 5],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    ),
    "dw1d_k5_dilation16_stride1": partial(
        _make_conv,
        [1, 64, 1, 65],
        [64, 1, 1, 5],
        [1, 64, 1, 1],
        weight_range=(-0.4453125, 0.4453125),
        bias_range=(-0.4375, 0.44140625),
        dilations=[1, 16],
        group=64,
        kernel_shape=[1, 5],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    ),
    "dw_1x4_dilation_1x2_expanded_1x7": partial(
        _make_conv,
        [1, 64, 1, 10],
        [64, 1, 1, 4],
        [1, 64, 1, 4],
        weight_range=(-0.8984375, 0.8984375),
        bias_range=(-0.2001953125, 0.30078125),
        dilations=[1, 2],
        group=64,
        kernel_shape=[1, 4],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    ),
    "dw_bf16_f5x2_s2x1_2x256x2": partial(
        _make_conv,
        [1, 2, 256, 2],
        [2, 1, 5, 2],
        [1, 2, 128, 1],
        weight_range=(-0.095703125, 0.07080078125),
        bias_value=0.0,
        dilations=[1, 1],
        group=2,
        kernel_shape=[5, 2],
        pads=[2, 0, 2, 0],
        strides=[2, 1],
    ),
    "conv_transpose_bf16_stride_2x2_interleave": partial(
        _make_conv,
        [1, 8, 4, 16],
        [8, 8, 2, 2],
        [1, 8, 8, 32],
        weight_range=(-1.625, 1.125),
        op="ConvTranspose",
        kernel_shape=[2, 2],
        pads=[0, 0, 0, 0],
        strides=[2, 2],
    ),
    "conv_transpose_bf16_stride_5x2": partial(
        _make_conv,
        [1, 1, 16, 2],
        [1, 1, 5, 2],
        [1, 1, 32, 1],
        weight_range=(-0.062255859375, 0.07421875),
        bias_value=0.0,
        op="ConvTranspose",
        dilations=[1, 1],
        group=1,
        kernel_shape=[5, 2],
        output_padding=[1, 0],
        pads=[2, 1, 2, 1],
        strides=[2, 1],
    ),
    "convtranspose1d-s2-crop-bf16": make_convtranspose1d_s2_crop_bf16,
    'Conv2d_bf16_1x1x256x256': partial(
        _make_conv,
        [1, 256, 128, 1],
        [256, 256, 1, 1],
        [1, 256, 128, 1],
        bias_range=(-0.0625, 0.062255859375),
        weight_range=(-0.0625, 0.0625),
        dilations=[1, 1],
        group=1,
        kernel_shape=[1, 1],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    ),
    'Conv2d_bf16_1x1x2x256': partial(
        _make_conv,
        [1, 2, 128, 1],
        [256, 2, 1, 1],
        [1, 256, 128, 1],
        weight_range=(-0.1455078125, 0.1552734375),
        bias_value=0.0,
        dilations=[1, 1],
        group=1,
        kernel_shape=[1, 1],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    ),
    'Conv2d_bf16_1x1x64x8192': partial(
        _make_conv,
        [1, 64, 1, 1],
        [8192, 64, 1, 1],
        [1, 8192, 1, 1],
        bias_range=(-0.125, 0.125),
        weight_range=(-0.125, 0.125),
        dilations=[1, 1],
        group=1,
        kernel_shape=[1, 1],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    ),
    'conv2d-relu': partial(
        _make_conv,
        [1, 192, 35, 35],
        [64, 192, 1, 1],
        [1, 64, 35, 35],
        bias_range=(-30.875, 11.625),
        opset=18,
        weight_range=(-0.80859375, 2.46875),
        activation='relu',
        dilations=[1, 1],
        group=1,
        kernel_shape=[1, 1],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    ),
    'qdq-conv2d': make_qdq_conv2d,
    'ConvTranspose_bf16_1x1x512x512': partial(
        _make_conv,
        [1, 512, 16, 1],
        [512, 512, 1, 1],
        [1, 512, 16, 1],
        op='ConvTranspose',
        weight_range=(-0.2255859375, 0.2412109375),
        bias_value=0.0,
        dilations=[1, 1],
        group=1,
        kernel_shape=[1, 1],
        pads=[0, 0, 0, 0],
        strides=[1, 1],
    ),

    'conv2d-host': partial(
        _make_conv,
        [1, 3, 224, 224],
        [64, 3, 11, 11],
        [1, 64, 55, 55],
        bias_range=(-2.796875, 1.640625),
        bias_first=True,
        opset=16,
        weight_range=(-0.78125, 0.93359375),
        dilations=[1, 1],
        group=1,
        kernel_shape=[11, 11],
        pads=[2, 2, 2, 2],
        strides=[4, 4],
    ),
    'conv1d-bf16-k127-s64': partial(
        _make_conv,
        [1, 1, 191],
        [288, 1, 127],
        [1, 288, 2],
        weight_range=(-1.6796875, 1.71875),
        auto_pad='NOTSET',
        dilations=[1],
        group=1,
        kernel_shape=[127],
        pads=[0, 0],
        strides=[64],
    ),
    'conv1d-bf16-c288-k7-s3': partial(
        _make_conv,
        [1, 288, 7],
        [576, 288, 7],
        [1, 576, 1],
        bias_range=(-1.3515625, 0.494140625),
        weight_range=(-2.171875, 1.8671875),
        auto_pad='NOTSET',
        dilations=[1],
        group=1,
        kernel_shape=[7],
        pads=[0, 0],
        strides=[3],
    ),
    'conv1d-bf16-c576-k3-s2': partial(
        _make_conv,
        [1, 576, 5],
        [288, 576, 3],
        [1, 288, 2],
        bias_range=(-0.77734375, 3.390625),
        weight_range=(-3.8125, 1.203125),
        auto_pad='NOTSET',
        dilations=[1],
        group=1,
        kernel_shape=[3],
        pads=[0, 0],
        strides=[2],
    ),
}
