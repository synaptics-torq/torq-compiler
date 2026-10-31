"""Concat operator fixtures built with torq.testing.onnx_builder."""

from functools import partial

from torq.testing.onnx_builder import F32, OnnxGraph

KIND = "onnx"


def _make_concat(input_0_shape, axis):
    g = OnnxGraph(opset=22)
    a = g.input("input_0", F32, input_0_shape)
    b = g.input("input_1", F32, input_0_shape)
    shape = list(input_0_shape)
    shape[axis] *= 2
    concat = g.op("Concat", g.op("Cast", a, to=16), g.op("Cast", b, to=16), axis=axis)
    return g.export(g.op("Cast", concat, to=1, dtype=F32, shape=shape))


CASES = {
    "concat-axis-2-3x3x2-b16": partial(_make_concat, [3, 3, 1], 2),
    "concat-axis-3-1x256x128x2-b16": partial(_make_concat, [1, 256, 128, 1], 3),
}
