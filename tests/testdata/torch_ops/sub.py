"""Sub operator fixtures built with torq.testing.onnx_builder."""

from functools import partial

import numpy as np

from torq.testing.onnx_builder import BF16, I32, OnnxGraph

KIND = "onnx"


def _make_sub(input_0_shape, input_1_shape, *, dtype=I32, opset=19):
    g = OnnxGraph(opset)
    a = g.input("input_0", dtype, input_0_shape)
    b = g.input("input_1", dtype, input_1_shape)
    shape = np.broadcast_shapes(input_0_shape, input_1_shape)
    return g.export(g.op("Sub", a, b, dtype=dtype, shape=shape))


def _make_sub_constant(shape, value, *, constant_first=False):
    """x - value, or value - x with constant_first, with value a bf16 constant of shape
    [1, 1, 1]."""
    g = OnnxGraph(opset=22)
    x = g.input("x", BF16, shape)
    c = g.const(np.full([1, 1, 1], value), BF16)
    a, b = (c, x) if constant_first else (x, c)
    return g.export(g.op("Sub", a, b, dtype=BF16, shape=shape))


CASES = {
    "scalar_into_higher_rank_midunit": partial(_make_sub, [], [2, 1, 4]),
    "sub_scalar_tensor_nss_i32": partial(_make_sub, [], [16010]),
    "tensor_minus_scalar_rhs": partial(_make_sub, [1, 256], []),
    # Ported from tests/testdata/tosa_ops, with the names of the MLIR files
    "sub-bf16": partial(_make_sub, [1, 56, 56, 24], [1, 56, 56, 24], dtype=BF16, opset=22),
    "sub-constant-bf16": partial(_make_sub_constant, [1, 21, 1024], 32768),
    "sub-scalar-tensor-bf16": partial(
        _make_sub_constant, [1, 21, 1024], 32768, constant_first=True
    ),
}
