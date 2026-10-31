"""Qdq operator fixtures built with ONNX GraphSurgeon."""

import numpy as np
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def make_qdq_mul() -> onnx.ModelProto:
    input_0 = gs.Variable("input_0", dtype=np.int8, shape=[1, 32, 112, 112])
    input_1 = gs.Variable("input_1", dtype=np.int8, shape=[1, 32, 112, 112])
    output = gs.Variable("output", dtype=np.int8, shape=[1, 32, 112, 112])
    zero_point = gs.Constant("zero_point", np.full([], 0, dtype=np.int8))
    scale = gs.Constant("scale", np.full([], 0.007843135856091976, dtype=np.float32))
    zero_point_2 = gs.Constant("zero_point_2", np.full([], -1, dtype=np.int8))
    scale_2 = gs.Constant("scale_2", np.full([], 0.00784313678741455, dtype=np.float32))
    zero_point_3 = gs.Constant("zero_point_3", np.full([], -1, dtype=np.int8))
    scale_3 = gs.Constant("scale_3", np.full([], 0.007838194258511066, dtype=np.float32))
    dequantizelinear = gs.Variable("dequantizelinear")
    dequantizelinear_2 = gs.Variable("dequantizelinear_2")
    mul = gs.Variable("mul")
    graph = gs.Graph(
        nodes=[
            gs.Node(
                "DequantizeLinear",
                inputs=[input_0, scale_2, zero_point_2],
                outputs=[dequantizelinear],
            ),
            gs.Node(
                "DequantizeLinear",
                inputs=[input_1, scale, zero_point],
                outputs=[dequantizelinear_2],
            ),
            gs.Node("Mul", inputs=[dequantizelinear_2, dequantizelinear], outputs=[mul]),
            gs.Node("QuantizeLinear", inputs=[mul, scale_3, zero_point_3], outputs=[output]),
        ],
        inputs=[input_0, input_1],
        outputs=[output],
        opset=17,
    )
    return gs.export_onnx(graph, ir_version=10)


def make_qdq_mul_broadcast_const_scale_1ch_remainder() -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    x = gs.Variable("x", dtype=np.int8, shape=[1, 9, 112, 112])
    output = gs.Variable("output", dtype=np.int8, shape=[1, 9, 112, 112])
    zero_point = gs.Constant("zero_point", np.full([], 0, dtype=np.int8))
    scale = gs.Constant("scale", np.full([], 0.007843135856091976, dtype=np.float32))
    zero_point_2 = gs.Constant("zero_point_2", np.full([], -1, dtype=np.int8))
    scale_2 = gs.Constant("scale_2", np.full([], 0.00784313678741455, dtype=np.float32))
    zero_point_3 = gs.Constant("zero_point_3", np.full([], -1, dtype=np.int8))
    scale_3 = gs.Constant("scale_3", np.full([], 0.007838194258511066, dtype=np.float32))
    quantized = gs.Constant("quantized", rng.integers(-50, 7, [1, 9, 1, 1], dtype=np.int8))
    dequantizelinear = gs.Variable("dequantizelinear")
    dequantizelinear_2 = gs.Variable("dequantizelinear_2")
    mul = gs.Variable("mul")
    graph = gs.Graph(
        nodes=[
            gs.Node(
                "DequantizeLinear", inputs=[x, scale_2, zero_point_2], outputs=[dequantizelinear]
            ),
            gs.Node(
                "DequantizeLinear",
                inputs=[quantized, scale, zero_point],
                outputs=[dequantizelinear_2],
            ),
            gs.Node("Mul", inputs=[dequantizelinear, dequantizelinear_2], outputs=[mul]),
            gs.Node("QuantizeLinear", inputs=[mul, scale_3, zero_point_3], outputs=[output]),
        ],
        inputs=[x],
        outputs=[output],
        opset=17,
    )
    return gs.export_onnx(graph, ir_version=10)


def make_qdq_mul_scalar_const() -> onnx.ModelProto:
    x = gs.Variable("x", dtype=np.int8, shape=[1, 32, 112, 112])
    output = gs.Variable("output", dtype=np.int8, shape=[1, 32, 112, 112])
    zero_point = gs.Constant("zero_point", np.full([], 0, dtype=np.int8))
    scale = gs.Constant("scale", np.full([], 0.007843135856091976, dtype=np.float32))
    zero_point_2 = gs.Constant("zero_point_2", np.full([], -1, dtype=np.int8))
    scale_2 = gs.Constant("scale_2", np.full([], 0.00784313678741455, dtype=np.float32))
    zero_point_3 = gs.Constant("zero_point_3", np.full([], -1, dtype=np.int8))
    scale_3 = gs.Constant("scale_3", np.full([], 0.007838194258511066, dtype=np.float32))
    quantized = gs.Constant("quantized", np.full([], 37, dtype=np.int8))
    dequantizelinear = gs.Variable("dequantizelinear")
    dequantizelinear_2 = gs.Variable("dequantizelinear_2")
    mul = gs.Variable("mul")
    graph = gs.Graph(
        nodes=[
            gs.Node(
                "DequantizeLinear", inputs=[x, scale_2, zero_point_2], outputs=[dequantizelinear]
            ),
            gs.Node(
                "DequantizeLinear",
                inputs=[quantized, scale, zero_point],
                outputs=[dequantizelinear_2],
            ),
            gs.Node("Mul", inputs=[dequantizelinear, dequantizelinear_2], outputs=[mul]),
            gs.Node("QuantizeLinear", inputs=[mul, scale_3, zero_point_3], outputs=[output]),
        ],
        inputs=[x],
        outputs=[output],
        opset=17,
    )
    return gs.export_onnx(graph, ir_version=10)


def make_qdq_sigmoid() -> onnx.ModelProto:
    x = gs.Variable("x", dtype=np.int8, shape=[1, 32, 112, 112])
    output = gs.Variable("output", dtype=np.int8, shape=[1, 32, 112, 112])
    zero_point = gs.Constant("zero_point", np.full([], 0, dtype=np.int8))
    scale = gs.Constant("scale", np.full([], 0.007843135856091976, dtype=np.float32))
    zero_point_2 = gs.Constant("zero_point_2", np.full([], -128, dtype=np.int8))
    scale_2 = gs.Constant("scale_2", np.full([], 0.0028668963350355625, dtype=np.float32))
    dequantizelinear = gs.Variable("dequantizelinear")
    sigmoid = gs.Variable("sigmoid")
    graph = gs.Graph(
        nodes=[
            gs.Node("DequantizeLinear", inputs=[x, scale, zero_point], outputs=[dequantizelinear]),
            gs.Node("Sigmoid", inputs=[dequantizelinear], outputs=[sigmoid]),
            gs.Node("QuantizeLinear", inputs=[sigmoid, scale_2, zero_point_2], outputs=[output]),
        ],
        inputs=[x],
        outputs=[output],
        opset=17,
    )
    return gs.export_onnx(graph, ir_version=10)


def make_qdq_min_i8() -> onnx.ModelProto:
    x = gs.Variable("x", dtype=np.float32, shape=[1, 64, 64])
    output = gs.Variable("output", dtype=np.float32, shape=[1, 64, 64])
    scale = gs.Constant("scale", np.full([], 0.007843137718737125, dtype=np.float32))
    zero_point = gs.Constant("zero_point", np.full([], -7, dtype=np.int8))
    quantizelinear = gs.Variable("quantizelinear")
    graph = gs.Graph(
        nodes=[
            gs.Node("QuantizeLinear", inputs=[x, scale, zero_point], outputs=[quantizelinear]),
            gs.Node(
                "DequantizeLinear", inputs=[quantizelinear, scale, zero_point], outputs=[output]
            ),
        ],
        inputs=[x],
        outputs=[output],
        opset=19,
    )
    return gs.export_onnx(graph, ir_version=10)


def make_qdq_mul_broadcast_const_scale() -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    x = gs.Variable("x", dtype=np.int8, shape=[1, 32, 112, 112])
    dequantized_x = gs.Variable("dequantized_x", dtype=np.float32, shape=[1, 32, 112, 112])
    dequantized_y = gs.Variable("dequantized_y", dtype=np.float32, shape=[1, 32, 1, 1])
    product = gs.Variable("product", dtype=np.float32, shape=[1, 32, 112, 112])
    output = gs.Variable("output", dtype=np.int8, shape=[1, 32, 112, 112])
    y_zero_point = gs.Constant("y_zero_point", np.full([], 0, dtype=np.int8))
    y_scale = gs.Constant("y_scale", np.full([], 0.007843135856091976, dtype=np.float32))
    x_zero_point = gs.Constant("x_zero_point", np.full([], -1, dtype=np.int8))
    x_scale = gs.Constant("x_scale", np.full([], 0.00784313678741455, dtype=np.float32))
    output_zero_point = gs.Constant("output_zero_point", np.full([], -1, dtype=np.int8))
    output_scale = gs.Constant("output_scale", np.full([], 0.007838194258511066, dtype=np.float32))
    quantized_scale = gs.Constant("quantized_scale", rng.integers(-50, 49, [1, 32, 1, 1], dtype=np.int8))
    graph = gs.Graph(
        nodes=[
            gs.Node("DequantizeLinear", inputs=[x, x_scale, x_zero_point], outputs=[dequantized_x]),
            gs.Node(
                "DequantizeLinear",
                inputs=[quantized_scale, y_scale, y_zero_point],
                outputs=[dequantized_y],
            ),
            gs.Node("Mul", inputs=[dequantized_x, dequantized_y], outputs=[product]),
            gs.Node(
                "QuantizeLinear",
                inputs=[product, output_scale, output_zero_point],
                outputs=[output],
            ),
        ],
        inputs=[x],
        outputs=[output],
        opset=17,
    )
    return gs.export_onnx(graph, ir_version=10)


def make_qdq_mul_broadcast() -> onnx.ModelProto:
    x = gs.Variable("x", dtype=np.int8, shape=[1, 32, 112, 112])
    y = gs.Variable("y", dtype=np.int8, shape=[1, 32, 1, 1])
    dequantized_x = gs.Variable("dequantized_x", dtype=np.float32, shape=[1, 32, 112, 112])
    dequantized_y = gs.Variable("dequantized_y", dtype=np.float32, shape=[1, 32, 1, 1])
    product = gs.Variable("product", dtype=np.float32, shape=[1, 32, 112, 112])
    output = gs.Variable("output", dtype=np.int8, shape=[1, 32, 112, 112])
    y_zero_point = gs.Constant("y_zero_point", np.full([], 0, dtype=np.int8))
    y_scale = gs.Constant("y_scale", np.full([], 0.007843135856091976, dtype=np.float32))
    x_zero_point = gs.Constant("x_zero_point", np.full([], -1, dtype=np.int8))
    x_scale = gs.Constant("x_scale", np.full([], 0.00784313678741455, dtype=np.float32))
    output_zero_point = gs.Constant("output_zero_point", np.full([], -1, dtype=np.int8))
    output_scale = gs.Constant("output_scale", np.full([], 0.007838194258511066, dtype=np.float32))
    graph = gs.Graph(
        nodes=[
            gs.Node("DequantizeLinear", inputs=[x, x_scale, x_zero_point], outputs=[dequantized_x]),
            gs.Node("DequantizeLinear", inputs=[y, y_scale, y_zero_point], outputs=[dequantized_y]),
            gs.Node("Mul", inputs=[dequantized_x, dequantized_y], outputs=[product]),
            gs.Node(
                "QuantizeLinear",
                inputs=[product, output_scale, output_zero_point],
                outputs=[output],
            ),
        ],
        inputs=[x, y],
        outputs=[output],
        opset=17,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "qdq-mul-broadcast-const-scale": make_qdq_mul_broadcast_const_scale,
    "qdq-mul-broadcast": make_qdq_mul_broadcast,
    "qdq-mul": make_qdq_mul,
    "qdq-mul-broadcast-const-scale-1ch-remainder": make_qdq_mul_broadcast_const_scale_1ch_remainder,
    "qdq-mul-scalar-const": make_qdq_mul_scalar_const,
    "qdq-sigmoid": make_qdq_sigmoid,
    "qdq_min_i8": make_qdq_min_i8,
}
