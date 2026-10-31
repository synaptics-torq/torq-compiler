"""ONNX operator fixtures built with GraphSurgeon."""

import numpy as np

from torq.testing.onnx_builder import I64, OnnxGraph

KIND = "onnx"


def make_equal():
    g = OnnxGraph(opset=22)
    x = g.input("x", I64, [1])
    time_ids = g.const(np.arange(30).reshape([1, 1, 30, 1]), I64)
    return g.export(g.op("Equal", time_ids, x, dtype=np.bool_, shape=[1, 1, 30, 1]))


CASES = {
    "equal": make_equal,
}
