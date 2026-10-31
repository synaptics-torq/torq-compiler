"""Dequantize operator fixtures built with torq.testing.onnx_builder."""

from torq.testing.onnx_builder import F32, I8, OnnxGraph

KIND = "onnx"


def make_deq_min_i8():
    g = OnnxGraph(opset=19)
    x = g.input("x", I8, [1, 64, 64])
    return g.export(g.dequantize(x, 0.007843137718737125, -7, dtype=F32, shape=[1, 64, 64]))


CASES = {
    "deq_min_i8": make_deq_min_i8,
}

CONFIG = {
    # int8 dequant zero-point regression: the residual is bf16 MulOp rounding.
    "deq_min_i8": {"comparison": {"fp_max_tol": 0.2, "allowed_wrong": 0.06}},
}
