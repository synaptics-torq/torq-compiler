"""Expand operator fixtures built with torq.testing.onnx_builder."""

from functools import partial

import numpy as np

from torq.testing.onnx_builder import BF16, I64, OnnxGraph

KIND = "onnx"


def _make_expand(input_shape, *, dtype=BF16):
    g = OnnxGraph(opset=22)
    x = g.input("x", dtype, input_shape)
    shape = [1, 8, 30, 36]
    return g.export(g.op("Expand", x, g.const(shape, I64), dtype=dtype, shape=shape))


CASES = {
    "expand_bf16_d2": partial(_make_expand, [1, 8, 1, 36]),
    "expand_i1_d1_d3": partial(_make_expand, [1, 1, 30, 1], dtype=np.bool_),
}
