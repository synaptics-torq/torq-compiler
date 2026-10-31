"""Mul operator fixtures built with torq.testing.onnx_builder."""

from torq.testing.onnx_builder import I64, OnnxGraph

KIND = "onnx"


def make_mul_i64_scalar():
    g = OnnxGraph(opset=18)
    a = g.input("input_0", I64, [2])
    b = g.input("input_1", I64, [])
    return g.export(g.op("Mul", a, b, dtype=I64, shape=[2]))


CASES = {
    "mul-i64-scalar": make_mul_i64_scalar,
}

CONFIG = {
    # The i64 scalar mul must run on the host, without slice lowering.
    "mul-i64-scalar": {"compiler_options": ["--torq-disable-slices", "--torq-disable-css"]},
}
