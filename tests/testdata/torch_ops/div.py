"""Div operator fixtures built with ONNX GraphSurgeon."""

import ml_dtypes
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def make_div_1x1x288xbf16() -> onnx.ModelProto:
    input_0 = gs.Variable("input_0", dtype=ml_dtypes.bfloat16, shape=[1, 1, 288])
    input_1 = gs.Variable("input_1", dtype=ml_dtypes.bfloat16, shape=[1, 1, 1])
    output = gs.Variable("output", dtype=ml_dtypes.bfloat16, shape=[1, 1, 288])
    graph = gs.Graph(
        nodes=[
            gs.Node("Div", inputs=[input_0, input_1], outputs=[output]),
        ],
        inputs=[input_0, input_1],
        outputs=[output],
        opset=22,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "div_1x1x288xbf16": make_div_1x1x288xbf16,
}
