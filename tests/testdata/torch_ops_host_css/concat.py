"""Concat operator fixtures built with torq.testing.onnx_builder."""

from torq.testing.onnx_builder import F32, OnnxGraph

KIND = "onnx"


def make_concat_axis_2_3x3x2_i8():
    g = OnnxGraph(opset=22)
    a = g.input("input_0", F32, [3, 3, 1])
    b = g.input("input_1", F32, [3, 3, 1])
    concat = g.op("Concat", g.op("Cast", a, to=3), g.op("Cast", b, to=3), axis=2)
    return g.export(g.op("Cast", concat, to=1, dtype=F32, shape=[3, 3, 2]))


CASES = {
    "concat-axis-2-3x3x2-i8": make_concat_axis_2_3x3x2_i8,
}
