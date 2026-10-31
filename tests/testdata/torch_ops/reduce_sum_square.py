"""ONNX operator fixtures built with GraphSurgeon."""

import ml_dtypes
import numpy as np
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def make_reducesumsquare_axes23() -> onnx.ModelProto:
    x = gs.Variable("x", dtype=ml_dtypes.bfloat16, shape=[1, 1280, 7, 7])
    output = gs.Variable("output", dtype=ml_dtypes.bfloat16, shape=[1, 1280, 1, 1])
    axes = gs.Constant("axes", np.array([-1, -2], dtype=np.int64))
    graph = gs.Graph(
        nodes=[
            gs.Node(
                "ReduceSumSquare",
                inputs=[x, axes],
                outputs=[output],
                attrs={"keepdims": 1, "noop_with_empty_axes": 0},
            ),
        ],
        inputs=[x],
        outputs=[output],
        opset=22,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "reducesumsquare-axes23": make_reducesumsquare_axes23,
}
