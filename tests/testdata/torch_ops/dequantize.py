"""Dequantize operator fixtures built with ONNX GraphSurgeon."""

import numpy as np
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def make_deq_min_i8() -> onnx.ModelProto:
    x = gs.Variable("x", dtype=np.int8, shape=[1, 64, 64])
    output = gs.Variable("output", dtype=np.float32, shape=[1, 64, 64])
    scale = gs.Constant("scale", np.full([], 0.007843137718737125, dtype=np.float32))
    zero_point = gs.Constant("zero_point", np.full([], -7, dtype=np.int8))
    graph = gs.Graph(
        nodes=[
            gs.Node("DequantizeLinear", inputs=[x, scale, zero_point], outputs=[output]),
        ],
        inputs=[x],
        outputs=[output],
        opset=19,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "deq_min_i8": make_deq_min_i8,
}
