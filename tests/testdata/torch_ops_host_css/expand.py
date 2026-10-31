"""Expand operator fixtures built with ONNX GraphSurgeon."""

import ml_dtypes
import numpy as np
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def make_i1_bf16() -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    x = gs.Variable("x", dtype=ml_dtypes.bfloat16, shape=[1, 3136, 64])
    output = gs.Variable("output", dtype=ml_dtypes.bfloat16, shape=[1, 3137, 64])
    # Mul/Equal/Where computes [1, 1, 1]; data propagation needs the Expand shape explicitly.
    expand = gs.Variable("expand", dtype=ml_dtypes.bfloat16, shape=[1, 1, 64])
    ones = gs.Constant("ones", np.full([3], 1, dtype=np.int64))
    target_shape = gs.Constant("target_shape", np.array([1, -1, -1], dtype=np.int64))
    negative_one = gs.Constant("negative_one", np.full([], -1, dtype=np.int64))
    class_token = gs.Constant(
        "class_token",
        rng.uniform(-0.294921875, 0.1689453125, [1, 1, 64]).astype(ml_dtypes.bfloat16),
    )
    mul = gs.Variable("mul")
    equal = gs.Variable("equal")
    where = gs.Variable("where")
    graph = gs.Graph(
        nodes=[
            gs.Node("Mul", inputs=[ones, negative_one], outputs=[mul]),
            gs.Node("Equal", inputs=[target_shape, mul], outputs=[equal]),
            gs.Node("Where", inputs=[equal, ones, target_shape], outputs=[where]),
            gs.Node("Expand", inputs=[class_token, where], outputs=[expand]),
            gs.Node("Concat", inputs=[expand, x], outputs=[output], attrs={"axis": 1}),
        ],
        inputs=[x],
        outputs=[output],
        opset=18,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "i1-bf16": make_i1_bf16,
}
