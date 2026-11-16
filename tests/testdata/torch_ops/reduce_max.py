"""ReduceMax operator fixtures built with torq.testing.onnx_builder.

The cases are ported from tests/testdata/tosa_ops, with the names of the MLIR files.
"""

from torq.testing.onnx_builder import F32, I64, OnnxGraph

KIND = "onnx"


def make_reduce_max_f32():
    g = OnnxGraph(opset=22)
    x = g.input("x", F32, [1, 1000])
    return g.export(g.op("ReduceMax", x, g.const([1], I64), keepdims=1, dtype=F32, shape=[1, 1]))


CASES = {
    "reduce-max-f32": make_reduce_max_f32,
}
