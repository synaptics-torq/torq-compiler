"""Transpose operator fixtures built with ONNX GraphSurgeon."""

from functools import partial

import ml_dtypes
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def _make_transpose(input_shape, perm) -> onnx.ModelProto:
    x = gs.Variable("x", dtype=ml_dtypes.bfloat16, shape=input_shape)
    output = gs.Variable("output", dtype=ml_dtypes.bfloat16, shape=[input_shape[i] for i in perm])
    graph = gs.Graph(
        nodes=[gs.Node("Transpose", inputs=[x], outputs=[output], attrs={"perm": perm})],
        inputs=[x],
        outputs=[output],
        opset=22,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "transpose-1x1x8192-bf16": partial(_make_transpose, [1, 1, 8192], [1, 2, 0]),
    "transpose-1x512x16x1-bf16": partial(_make_transpose, [1, 512, 16, 1], [3, 0, 1, 2]),
    "transpose-1x64x1x1-bf16": partial(_make_transpose, [1, 64, 1, 1], [0, 3, 2, 1]),
}
