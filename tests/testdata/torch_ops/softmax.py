"""Softmax operator fixtures built with torq.testing.onnx_builder."""

from functools import partial

from torq.testing.onnx_builder import BF16, OnnxGraph

KIND = "onnx"


def _make_softmax(shape, **attrs):
    g = OnnxGraph(opset=22)
    x = g.input("x", BF16, shape)
    return g.export(g.op("Softmax", x, dtype=BF16, shape=shape, **attrs))


CASES = {
    "softmax-1x8x1x207xbf16": partial(_make_softmax, [1, 8, 1, 207], axis=-1),
    "softmax-1x2xbf16": partial(_make_softmax, [1, 2]),
}
