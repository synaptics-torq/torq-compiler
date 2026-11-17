"""ReduceMax operator fixtures built with torq.testing.onnx_builder.

The cases are ported from tests/testdata/tosa_ops, with the names of the MLIR files.
"""

from functools import partial

from torq.testing.onnx_builder import F32, I32, I64, OnnxGraph

KIND = "onnx"


def _make_reduce_max(dtype):
    g = OnnxGraph(opset=22)
    x = g.input("x", dtype, [1, 1000])
    return g.export(g.op("ReduceMax", x, g.const([1], I64), keepdims=1, dtype=dtype, shape=[1, 1]))


CASES = {
    "reduce-max-f32": partial(_make_reduce_max, F32),
    "reduce-max-i32": partial(_make_reduce_max, I32),
}
