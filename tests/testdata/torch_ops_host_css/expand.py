"""Expand operator fixtures built with torq.testing.onnx_builder."""

import numpy as np

from torq.testing.onnx_builder import BF16, I64, OnnxGraph

KIND = "onnx"


def make_i1_bf16():
    rng = np.random.default_rng(0)
    g = OnnxGraph(opset=18)
    x = g.input("x", BF16, [1, 3136, 64])
    ones = g.const([1, 1, 1], I64)
    target_shape = g.const([1, -1, -1], I64)
    class_token = g.const(rng.uniform(-0.294921875, 0.1689453125, [1, 1, 64]), BF16)
    equal = g.op("Equal", target_shape, g.op("Mul", ones, g.const(-1, I64)))
    where = g.op("Where", equal, ones, target_shape)
    # Mul/Equal/Where computes [1, 1, 1]; data propagation needs the Expand shape explicitly.
    expand = g.op("Expand", class_token, where, dtype=BF16, shape=[1, 1, 64])
    return g.export(g.op("Concat", expand, x, axis=1, dtype=BF16, shape=[1, 3137, 64]))


CASES = {
    "i1-bf16": make_i1_bf16,
}
