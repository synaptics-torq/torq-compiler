"""Elementwise operator fixtures built with torq.testing.onnx_builder: unary ops, Clip, Mul and
comparisons.

The cases are ported from tests/testdata/tosa_ops, with the names of the MLIR files. The bf16 Max
and Min cases stay there: ONNX Max and Min lower to a compare and a select, which the slices
don't support (synaptics-torq/torq-compiler-dev#3061), while TOSA maximum and minimum lower to
arith.maximumf and minimumf.
"""

from functools import partial

import numpy as np

from torq.testing.onnx_builder import BF16, F32, OnnxGraph

KIND = "onnx"


def _make_unary(op, shape, *, dtype=BF16):
    g = OnnxGraph(opset=22)
    x = g.input("x", dtype, shape)
    return g.export(g.op(op, x, dtype=dtype, shape=shape))


def _make_binary(op, input_0_shape, input_1_shape, *, dtype=BF16, out_dtype=None):
    g = OnnxGraph(opset=22)
    a = g.input("input_0", dtype, input_0_shape)
    b = g.input("input_1", dtype, input_1_shape)
    shape = np.broadcast_shapes(input_0_shape, input_1_shape)
    return g.export(g.op(op, a, b, dtype=out_dtype or dtype, shape=shape))


def _make_clip(shape, low, high, *, dtype=F32):
    g = OnnxGraph(opset=22)
    x = g.input("x", dtype, shape)
    return g.export(g.op("Clip", x, g.const(low, dtype), g.const(high, dtype), dtype=dtype,
                         shape=shape))


CASES = {
    "abs-bf16": partial(_make_unary, "Abs", [1, 2, 3, 6, 65]),
    "abs-fp32": partial(_make_unary, "Abs", [1, 65, 4, 3, 6], dtype=F32),
    "ceil-bf16": partial(_make_unary, "Ceil", [1, 13, 21, 3]),
    "ceil-fp32": partial(_make_unary, "Ceil", [13, 21, 3], dtype=F32),
    "floor-bf16": partial(_make_unary, "Floor", [1, 13, 21, 3, 2]),
    "floor-fp32": partial(_make_unary, "Floor", [13, 21, 3], dtype=F32),
    "negate-bf16": partial(_make_unary, "Neg", [130, 6, 3]),
    "negate-fp32": partial(_make_unary, "Neg", [1300], dtype=F32),
    "clamp-fp32": partial(_make_clip, [1, 24, 3], 0.0, 6.0),
    "mul-bf16": partial(_make_binary, "Mul", [1, 68, 68, 7], [1, 68, 68, 7]),
    # The second input is broadcast along the second axis
    "mul-scalar": partial(_make_binary, "Mul", [1, 8, 207, 32], [1, 1, 207, 32]),
    "Elementwise_eq-bf16": partial(_make_binary, "Equal", [4, 64], [4, 64], out_dtype=np.bool_),
    "Elementwise_gt-bf16": partial(_make_binary, "Greater", [4, 64], [4, 64], out_dtype=np.bool_),
    "Elementwise_gteq-bf16": partial(
        _make_binary, "GreaterOrEqual", [4, 64], [4, 64], out_dtype=np.bool_
    ),
}
