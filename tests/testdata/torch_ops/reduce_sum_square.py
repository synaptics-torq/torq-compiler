"""ONNX operator fixtures built with GraphSurgeon."""

from torq.testing.onnx_builder import BF16, I64, OnnxGraph

KIND = "onnx"


def make_reducesumsquare_axes23():
    g = OnnxGraph(opset=22)
    x = g.input("x", BF16, [1, 1280, 7, 7])
    axes = g.const([-1, -2], I64)
    y = g.op("ReduceSumSquare", x, axes, keepdims=1, noop_with_empty_axes=0, dtype=BF16,
             shape=[1, 1280, 1, 1])
    return g.export(y)


CASES = {
    "reducesumsquare-axes23": make_reducesumsquare_axes23,
}
