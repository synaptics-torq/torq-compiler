"""Add operator fixtures built with torq.testing.onnx_builder."""

from functools import partial

import numpy as np

from torq.testing.onnx_builder import BF16, I32, OnnxGraph

KIND = "onnx"


def _make_add(input_0_shape, input_1_shape, *, dtype=BF16, opset=22):
    g = OnnxGraph(opset)
    a = g.input("input_0", dtype, input_0_shape)
    b = g.input("input_1", dtype, input_1_shape)
    shape = np.broadcast_shapes(input_0_shape, input_1_shape)
    return g.export(g.op("Add", a, b, dtype=dtype, shape=shape))


def _make_add_constant(shape, value, *, dtype=I32):
    """x + value, with value an i32 constant of shape [1, 1, 1]."""
    g = OnnxGraph(opset=22)
    x = g.input("x", dtype, shape)
    return g.export(g.op("Add", x, g.const(np.full([1, 1, 1], value), dtype), dtype=dtype,
                         shape=shape))


CASES = {
    "add-1x24x56x56-bf16": partial(_make_add, [1, 24, 56, 56], [1, 24, 56, 56]),
    "add-25x511-bf16": partial(_make_add, [25, 511], [25, 511]),
    "add-9x511-bf16": partial(_make_add, [9, 511], [9, 511]),
    "add-nss-25x511-bf16": partial(_make_add, [25, 511], [25, 511]),
    "add_scalar_into_unit_dim": partial(_make_add, [], [1, 256], dtype=np.int32, opset=19),
    # Ported from tests/testdata/tosa_ops, with the names of the MLIR files
    "add-bf16": partial(_make_add, [1, 56, 56, 24], [1, 56, 56, 24]),
    "Elementwise_add-i32": partial(_make_add, [4, 64, 64], [4, 64, 64], dtype=I32),
    "add-i32": partial(_make_add, [1, 56, 56, 24], [1, 56, 56, 24], dtype=I32),
    "add-i32-3d": partial(_make_add, [1, 21, 1024], [1, 21, 1024], dtype=I32),
    "add-broadcast": partial(_make_add, [1, 21, 1024], [1, 1, 1], dtype=I32),
    # add-constant and add-strided-scalar-i32, which were the same MLIR
    "add-constant-1x21x1024-i32": partial(_make_add_constant, [1, 21, 1024], 32767),
}
