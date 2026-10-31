"""Sqrt operator fixtures built with ONNX GraphSurgeon."""

from functools import partial

import ml_dtypes
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def _make_sqrt(input_shape) -> onnx.ModelProto:
    x = gs.Variable("x", dtype=ml_dtypes.bfloat16, shape=input_shape)
    output = gs.Variable("output", dtype=ml_dtypes.bfloat16, shape=input_shape)
    graph = gs.Graph(
        nodes=[gs.Node("Sqrt", inputs=[x], outputs=[output])],
        inputs=[x],
        outputs=[output],
        opset=22,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "sqrt-1x207x1-bf16": partial(_make_sqrt, [1, 207, 1]),
    "sqrt-scalar-bf16": partial(_make_sqrt, [1, 1, 1, 1]),
}
