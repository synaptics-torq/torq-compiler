"""Concat operator fixtures built with ONNX GraphSurgeon."""

from functools import partial

import numpy as np
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def _make_concat(input_0_shape, axis) -> onnx.ModelProto:
    input_0 = gs.Variable("input_0", dtype=np.float32, shape=input_0_shape)
    input_1 = gs.Variable("input_1", dtype=np.float32, shape=input_0_shape)
    output_shape = list(input_0_shape)
    output_shape[axis] *= 2
    output = gs.Variable("output", dtype=np.float32, shape=output_shape)
    cast = gs.Variable("cast")
    cast_2 = gs.Variable("cast_2")
    concat = gs.Variable("concat")
    graph = gs.Graph(
        nodes=[
            gs.Node("Cast", inputs=[input_0], outputs=[cast], attrs={"to": 16}),
            gs.Node("Cast", inputs=[input_1], outputs=[cast_2], attrs={"to": 16}),
            gs.Node("Concat", inputs=[cast, cast_2], outputs=[concat], attrs={"axis": axis}),
            gs.Node("Cast", inputs=[concat], outputs=[output], attrs={"to": 1}),
        ],
        inputs=[input_0, input_1],
        outputs=[output],
        opset=22,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "concat-axis-2-3x3x2-b16": partial(_make_concat, [3, 3, 1], 2),
    "concat-axis-3-1x256x128x2-b16": partial(_make_concat, [1, 256, 128, 1], 3),
}
