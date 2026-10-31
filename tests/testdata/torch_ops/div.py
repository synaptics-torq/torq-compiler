"""Div operator fixtures built with torq.testing.onnx_builder."""

from torq.testing.onnx_builder import BF16, OnnxGraph

KIND = "onnx"


def make_div_1x1x288xbf16():
    g = OnnxGraph(opset=22)
    a = g.input("input_0", BF16, [1, 1, 288])
    b = g.input("input_1", BF16, [1, 1, 1])
    return g.export(g.op("Div", a, b, dtype=BF16, shape=[1, 1, 288]))


CASES = {
    "div_1x1x288xbf16": make_div_1x1x288xbf16,
}
