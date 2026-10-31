"""Concat operator fixtures built with ONNX GraphSurgeon."""

import numpy as np
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def make_concat_axis_2_3x3x2_i8() -> onnx.ModelProto:
    input_0 = gs.Variable("input_0", dtype=np.float32, shape=[3, 3, 1])
    input_1 = gs.Variable("input_1", dtype=np.float32, shape=[3, 3, 1])
    output = gs.Variable("output", dtype=np.float32, shape=[3, 3, 2])
    cast = gs.Variable("cast")
    cast_2 = gs.Variable("cast_2")
    concat = gs.Variable("concat")
    graph = gs.Graph(
        nodes=[
            gs.Node("Cast", inputs=[input_0], outputs=[cast], attrs={"to": 3}),
            gs.Node("Cast", inputs=[input_1], outputs=[cast_2], attrs={"to": 3}),
            gs.Node("Concat", inputs=[cast, cast_2], outputs=[concat], attrs={"axis": 2}),
            gs.Node("Cast", inputs=[concat], outputs=[output], attrs={"to": 1}),
        ],
        inputs=[input_0, input_1],
        outputs=[output],
        opset=22,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "concat-axis-2-3x3x2-i8": make_concat_axis_2_3x3x2_i8,
}
