"""ONNX operator fixtures built with GraphSurgeon."""

import numpy as np
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def make_qoperator_mul() -> onnx.ModelProto:
    x = gs.Variable("x", dtype=np.int8, shape=[1, 32, 112, 112])
    y = gs.Variable("y", dtype=np.int8, shape=[1, 32, 112, 112])
    output = gs.Variable("output", dtype=np.int8, shape=[1, 32, 112, 112])
    output_zero_point = gs.Constant("output_zero_point", np.full([], -1, dtype=np.int8))
    output_scale = gs.Constant("output_scale", np.full([], 0.007838194258511066, dtype=np.float32))
    y_zero_point = gs.Constant("y_zero_point", np.full([], 0, dtype=np.int8))
    y_scale = gs.Constant("y_scale", np.full([], 0.007843135856091976, dtype=np.float32))
    x_zero_point = gs.Constant("x_zero_point", np.full([], -1, dtype=np.int8))
    x_scale = gs.Constant("x_scale", np.full([], 0.00784313678741455, dtype=np.float32))
    graph = gs.Graph(
        nodes=[
            gs.Node(
                "QLinearMul",
                inputs=[y, y_scale, y_zero_point, x, x_scale, x_zero_point, output_scale, output_zero_point],
                outputs=[output],
                domain="com.microsoft",
            ),
        ],
        inputs=[x, y],
        outputs=[output],
        opset=17,
    )
    graph.import_domains = [
        onnx.helper.make_opsetid("", 17),
        onnx.helper.make_opsetid("com.microsoft", 1),
    ]
    return gs.export_onnx(graph, ir_version=10)


def make_qoperator_sigmoid() -> onnx.ModelProto:
    x = gs.Variable("x", dtype=np.int8, shape=[1, 32, 112, 112])
    output = gs.Variable("output", dtype=np.int8, shape=[1, 32, 112, 112])
    output_zero_point = gs.Constant("output_zero_point", np.full([], -128, dtype=np.int8))
    output_scale = gs.Constant("output_scale", np.full([], 0.0028668963350355625, dtype=np.float32))
    input_zero_point = gs.Constant("input_zero_point", np.full([], 0, dtype=np.int8))
    input_scale = gs.Constant("input_scale", np.full([], 0.007843135856091976, dtype=np.float32))
    graph = gs.Graph(
        nodes=[
            gs.Node(
                "QLinearSigmoid",
                inputs=[x, input_scale, input_zero_point, output_scale, output_zero_point],
                outputs=[output],
                domain="com.microsoft",
            ),
        ],
        inputs=[x],
        outputs=[output],
        opset=17,
    )
    graph.import_domains = [
        onnx.helper.make_opsetid("", 17),
        onnx.helper.make_opsetid("com.microsoft", 1),
    ]
    return gs.export_onnx(graph, ir_version=10)


def make_qoperator_conv2d() -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    x = gs.Variable('x', dtype=np.int8, shape=[1, 3, 224, 224])
    output = gs.Variable('output', dtype=np.int8, shape=[1, 64, 112, 112])
    output_zero_point = gs.Constant('output_zero_point', np.full([], -22, dtype=np.int8))
    output_scale = gs.Constant('output_scale', np.full([], 0.13208869099617004, dtype=np.float32))
    input_zero_point = gs.Constant('input_zero_point', np.full([], -1, dtype=np.int8))
    input_scale = gs.Constant('input_scale', np.full([], 0.007843131199479103, dtype=np.float32))
    weight_scale = gs.Constant('weight_scale', np.full([], 0.022440632805228233, dtype=np.float32))
    weight_zero_point = gs.Constant('weight_zero_point', np.full([], 0, dtype=np.int8))
    weights = gs.Constant('weights', rng.integers(-127, 102, [64, 3, 7, 7], dtype=np.int8))
    bias = gs.Constant('bias', rng.integers(-30607, 25279, [64], dtype=np.int32))
    graph = gs.Graph(
        nodes=[
            gs.Node('QLinearConv', inputs=[x, input_scale, input_zero_point, weights, weight_scale, weight_zero_point, output_scale, output_zero_point, bias], outputs=[output], attrs={'dilations': [1, 1], 'group': 1, 'kernel_shape': [7, 7], 'pads': [3, 3, 3, 3], 'strides': [2, 2]}),
        ],
        inputs=[x],
        outputs=[output],
        opset=18,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "qoperator-mul": make_qoperator_mul,
    "qoperator-sigmoid": make_qoperator_sigmoid,
    'qoperator-conv2d': make_qoperator_conv2d,
}
