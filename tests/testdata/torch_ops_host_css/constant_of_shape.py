"""Constant of shape operator fixtures built with ONNX GraphSurgeon."""

import numpy as np
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def make_constantshape() -> onnx.ModelProto:
    output = gs.Variable("output", dtype=np.int64, shape=[2])
    constantofshape_constant = gs.Constant(
        "constantofshape_constant", np.full([1], 2, dtype=np.int64)
    )
    graph = gs.Graph(
        nodes=[
            gs.Node(
                "ConstantOfShape",
                inputs=[constantofshape_constant],
                outputs=[output],
                attrs={"value": gs.Constant("value", np.full([1], 1, dtype=np.int64))},
            ),
        ],
        inputs=[],
        outputs=[output],
        opset=18,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "constantshape": make_constantshape,
}
