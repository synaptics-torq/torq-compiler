"""ONNX operator fixtures built with GraphSurgeon."""

import numpy as np
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def make_equal() -> onnx.ModelProto:
    x = gs.Variable("x", dtype=np.int64, shape=[1])
    output = gs.Variable("output", dtype=np.bool_, shape=[1, 1, 30, 1])
    time_ids = gs.Constant("time_ids", np.arange(30, dtype=np.int64).reshape([1, 1, 30, 1]))
    graph = gs.Graph(
        nodes=[
            gs.Node("Equal", inputs=[time_ids, x], outputs=[output]),
        ],
        inputs=[x],
        outputs=[output],
        opset=22,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "equal": make_equal,
}
