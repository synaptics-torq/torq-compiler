"""Pow operator fixtures built with torq.testing.onnx_builder."""

from functools import partial

from torq.testing.onnx_builder import BF16, OnnxGraph

KIND = "onnx"


def _make_pow(input_shape, *, exponent=2.0):
    g = OnnxGraph(opset=22)
    x = g.input("x", BF16, input_shape)
    return g.export(g.op("Pow", x, g.const(exponent, BF16), dtype=BF16, shape=input_shape))


CASES = {
    "pow-half-scalar-bf16": partial(_make_pow, [1, 1], exponent=0.5),
    "pow2-1x207x288xbf16": partial(_make_pow, [1, 207, 288]),
    "pow2-scalar-bf16": partial(_make_pow, [1, 1, 1]),
}
