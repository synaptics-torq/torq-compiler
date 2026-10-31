"""Mul operator fixtures built with ONNX GraphSurgeon."""

import numpy as np
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def make_mul_i64_scalar() -> onnx.ModelProto:
    input_0 = gs.Variable("input_0", dtype=np.int64, shape=[2])
    input_1 = gs.Variable("input_1", dtype=np.int64, shape=[])
    output = gs.Variable("output", dtype=np.int64, shape=[2])
    graph = gs.Graph(
        nodes=[
            gs.Node("Mul", inputs=[input_0, input_1], outputs=[output]),
        ],
        inputs=[input_0, input_1],
        outputs=[output],
        opset=18,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "mul-i64-scalar": make_mul_i64_scalar,
}
