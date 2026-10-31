"""Sqrt operator fixtures built with torq.testing.onnx_builder."""

from functools import partial

from torq.testing.onnx_builder import BF16, OnnxGraph

KIND = "onnx"


def _make_sqrt(input_shape):
    g = OnnxGraph(opset=22)
    x = g.input("x", BF16, input_shape)
    return g.export(g.op("Sqrt", x, dtype=BF16, shape=input_shape))


CASES = {
    "sqrt-1x207x1-bf16": partial(_make_sqrt, [1, 207, 1]),
    "sqrt-scalar-bf16": partial(_make_sqrt, [1, 1, 1, 1]),
}

# The square root of a negative input is NaN.
CONFIG = {
    "sqrt-1x207x1-bf16": {"input_range": [0, 100]},
    "sqrt-scalar-bf16": {"input_range": [0, 100]},
}
