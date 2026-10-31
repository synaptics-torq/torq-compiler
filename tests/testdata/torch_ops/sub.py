"""Sub operator fixtures built with torq.testing.onnx_builder."""

from functools import partial

import numpy as np

from torq.testing.onnx_builder import I32, OnnxGraph

KIND = "onnx"


def _make_sub(input_0_shape, input_1_shape):
    g = OnnxGraph(opset=19)
    a = g.input("input_0", I32, input_0_shape)
    b = g.input("input_1", I32, input_1_shape)
    shape = np.broadcast_shapes(input_0_shape, input_1_shape)
    return g.export(g.op("Sub", a, b, dtype=I32, shape=shape))


CASES = {
    "scalar_into_higher_rank_midunit": partial(_make_sub, [], [2, 1, 4]),
    "sub_scalar_tensor_nss_i32": partial(_make_sub, [], [16010]),
    "tensor_minus_scalar_rhs": partial(_make_sub, [1, 256], []),
}
