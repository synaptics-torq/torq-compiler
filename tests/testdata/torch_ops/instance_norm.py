"""ONNX operator fixtures built with GraphSurgeon."""

import ml_dtypes
import numpy as np
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def make_instancenorm() -> onnx.ModelProto:
    x = gs.Variable("x", dtype=ml_dtypes.bfloat16, shape=[1, 1, 359712])
    output = gs.Variable("output", dtype=ml_dtypes.bfloat16, shape=[1, 1, 359712])
    scale = gs.Constant("scale", np.full([1], 1.0, dtype=ml_dtypes.bfloat16))
    bias = gs.Constant("bias", np.full([1], 0.0, dtype=ml_dtypes.bfloat16))
    graph = gs.Graph(
        nodes=[
            gs.Node(
                "InstanceNormalization",
                inputs=[x, scale, bias],
                outputs=[output],
                attrs={"epsilon": 9.99999974e-06},
            ),
        ],
        inputs=[x],
        outputs=[output],
        opset=22,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "instancenorm": make_instancenorm,
}
