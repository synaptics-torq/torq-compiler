"""Is nan operator fixtures built with ONNX GraphSurgeon."""

import ml_dtypes
import numpy as np
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def make_isnan_check_1x4x1x256() -> onnx.ModelProto:
    x = gs.Variable("x", dtype=ml_dtypes.bfloat16, shape=[1, 4, 1, 256])
    output = gs.Variable("output", dtype=ml_dtypes.bfloat16, shape=[1, 4, 1, 256])
    where_constant = gs.Constant("where_constant", np.full([1], 0.0, dtype=ml_dtypes.bfloat16))
    softmax = gs.Variable("softmax")
    isnan = gs.Variable("isnan")
    graph = gs.Graph(
        nodes=[
            gs.Node("Softmax", inputs=[x], outputs=[softmax]),
            gs.Node("IsNaN", inputs=[softmax], outputs=[isnan]),
            gs.Node("Where", inputs=[isnan, where_constant, softmax], outputs=[output]),
        ],
        inputs=[x],
        outputs=[output],
        opset=20,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "isnan_check-1x4x1x256": make_isnan_check_1x4x1x256,
}
