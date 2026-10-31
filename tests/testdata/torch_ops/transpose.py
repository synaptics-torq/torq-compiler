"""Transpose operator fixtures built with torq.testing.onnx_builder."""

from functools import partial

from torq.testing.onnx_builder import BF16, OnnxGraph

KIND = "onnx"


def _make_transpose(input_shape, perm):
    g = OnnxGraph(opset=22)
    x = g.input("x", BF16, input_shape)
    shape = [input_shape[i] for i in perm]
    return g.export(g.op("Transpose", x, perm=perm, dtype=BF16, shape=shape))


CASES = {
    "transpose-1x1x8192-bf16": partial(_make_transpose, [1, 1, 8192], [1, 2, 0]),
    "transpose-1x512x16x1-bf16": partial(_make_transpose, [1, 512, 16, 1], [3, 0, 1, 2]),
    "transpose-1x64x1x1-bf16": partial(_make_transpose, [1, 64, 1, 1], [0, 3, 2, 1]),
}
