"""Add operator fixtures built with torq.testing.onnx_builder."""

from functools import partial

import numpy as np

from torq.testing.onnx_builder import BF16, OnnxGraph

KIND = "onnx"


def _make_add(input_0_shape, input_1_shape, *, dtype=BF16, opset=22):
    g = OnnxGraph(opset)
    a = g.input("input_0", dtype, input_0_shape)
    b = g.input("input_1", dtype, input_1_shape)
    shape = np.broadcast_shapes(input_0_shape, input_1_shape)
    return g.export(g.op("Add", a, b, dtype=dtype, shape=shape))


CASES = {
    "add-1x24x56x56-bf16": partial(_make_add, [1, 24, 56, 56], [1, 24, 56, 56]),
    "add-25x511-bf16": partial(_make_add, [25, 511], [25, 511]),
    "add-9x511-bf16": partial(_make_add, [9, 511], [9, 511]),
    "add-nss-25x511-bf16": partial(_make_add, [25, 511], [25, 511]),
    "add_scalar_into_unit_dim": partial(_make_add, [], [1, 256], dtype=np.int32, opset=19),
}
