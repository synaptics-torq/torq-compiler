"""Add operator fixtures built with ONNX GraphSurgeon."""

from functools import partial

import ml_dtypes
import numpy as np
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def _make_add(
    input_0_shape, input_1_shape, *, dtype=ml_dtypes.bfloat16, opset=22
) -> onnx.ModelProto:
    input_0 = gs.Variable("input_0", dtype=dtype, shape=input_0_shape)
    input_1 = gs.Variable("input_1", dtype=dtype, shape=input_1_shape)
    output = gs.Variable(
        "output", dtype=dtype, shape=np.broadcast_shapes(input_0_shape, input_1_shape)
    )
    graph = gs.Graph(
        nodes=[gs.Node("Add", inputs=[input_0, input_1], outputs=[output])],
        inputs=[input_0, input_1],
        outputs=[output],
        opset=opset,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "add-1x24x56x56-bf16": partial(_make_add, [1, 24, 56, 56], [1, 24, 56, 56]),
    "add-25x511-bf16": partial(_make_add, [25, 511], [25, 511]),
    "add-9x511-bf16": partial(_make_add, [9, 511], [9, 511]),
    "add-nss-25x511-bf16": partial(_make_add, [25, 511], [25, 511]),
    "add_scalar_into_unit_dim": partial(_make_add, [], [1, 256], dtype=np.int32, opset=19),
}
