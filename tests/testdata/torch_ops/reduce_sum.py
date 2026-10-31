"""ONNX operator fixtures built with GraphSurgeon."""

from torq.testing.onnx_builder import F32, I64, OnnxGraph

KIND = "onnx"


def make_reducesum_axes23_keepdims():
    g = OnnxGraph(opset=22)
    x = g.input("x", F32, [1, 1280, 7, 7])
    axes = g.const([-1, -2], I64)
    y = g.op("ReduceSum", x, axes, keepdims=1, noop_with_empty_axes=0, dtype=F32,
             shape=[1, 1280, 1, 1])
    return g.export(y)


CASES = {
    "reducesum-axes23-keepdims": make_reducesum_axes23_keepdims,
}
