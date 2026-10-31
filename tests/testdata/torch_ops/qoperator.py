"""ONNX operator fixtures built with GraphSurgeon."""

import numpy as np

from torq.testing.onnx_builder import F32, I8, I32, OnnxGraph

KIND = "onnx"


def make_qoperator_mul():
    g = OnnxGraph(opset=17, domains={"com.microsoft": 1})
    x = g.input("x", I8, [1, 32, 112, 112])
    y = g.input("y", I8, [1, 32, 112, 112])
    inputs = [
        y, g.const(0.007843135856091976, F32), g.const(0, I8),
        x, g.const(0.00784313678741455, F32), g.const(-1, I8),
        g.const(0.007838194258511066, F32), g.const(-1, I8),
    ]
    out = g.op("QLinearMul", *inputs, domain="com.microsoft", dtype=I8, shape=[1, 32, 112, 112])
    return g.export(out)


def make_qoperator_sigmoid():
    g = OnnxGraph(opset=17, domains={"com.microsoft": 1})
    x = g.input("x", I8, [1, 32, 112, 112])
    inputs = [
        x, g.const(0.007843135856091976, F32), g.const(0, I8),
        g.const(0.0028668963350355625, F32), g.const(-128, I8),
    ]
    out = g.op("QLinearSigmoid", *inputs, domain="com.microsoft", dtype=I8,
               shape=[1, 32, 112, 112])
    return g.export(out)


def make_qoperator_conv2d():
    rng = np.random.default_rng(0)
    g = OnnxGraph(opset=18)
    x = g.input("x", I8, [1, 3, 224, 224])
    weights = g.const(rng.integers(-127, 102, [64, 3, 7, 7], dtype=np.int8), I8)
    bias = g.const(rng.integers(-30607, 25279, [64], dtype=np.int32), I32)
    inputs = [
        x, g.const(0.007843131199479103, F32), g.const(-1, I8),
        weights, g.const(0.022440632805228233, F32), g.const(0, I8),
        g.const(0.13208869099617004, F32), g.const(-22, I8),
        bias,
    ]
    out = g.op("QLinearConv", *inputs, dilations=[1, 1], group=1, kernel_shape=[7, 7],
               pads=[3, 3, 3, 3], strides=[2, 2], dtype=I8, shape=[1, 64, 112, 112])
    return g.export(out)


CASES = {
    "qoperator-mul": make_qoperator_mul,
    "qoperator-sigmoid": make_qoperator_sigmoid,
    'qoperator-conv2d': make_qoperator_conv2d,
}
