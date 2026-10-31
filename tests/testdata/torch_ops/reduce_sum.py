"""ONNX operator fixtures built with GraphSurgeon."""

import numpy as np
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def make_reducesum_axes23_keepdims() -> onnx.ModelProto:
    x = gs.Variable("x", dtype=np.float32, shape=[1, 1280, 7, 7])
    output = gs.Variable("output", dtype=np.float32, shape=[1, 1280, 1, 1])
    axes = gs.Constant("axes", np.array([-1, -2], dtype=np.int64))
    graph = gs.Graph(
        nodes=[
            gs.Node(
                "ReduceSum",
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
    "reducesum-axes23-keepdims": make_reducesum_axes23_keepdims,
}
