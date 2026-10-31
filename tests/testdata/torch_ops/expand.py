"""Expand operator fixtures built with ONNX GraphSurgeon."""

from functools import partial

import ml_dtypes
import numpy as np
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def _make_expand(input_shape, *, dtype=ml_dtypes.bfloat16) -> onnx.ModelProto:
    x = gs.Variable("x", dtype=dtype, shape=input_shape)
    output = gs.Variable("output", dtype=dtype, shape=[1, 8, 30, 36])
    shape = gs.Constant("shape", np.array([1, 8, 30, 36], dtype=np.int64))
    graph = gs.Graph(
        nodes=[gs.Node("Expand", inputs=[x, shape], outputs=[output])],
        inputs=[x],
        outputs=[output],
        opset=22,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "expand_bf16_d2": partial(_make_expand, [1, 8, 1, 36]),
    "expand_i1_d1_d3": partial(_make_expand, [1, 1, 30, 1], dtype=np.bool_),
}
