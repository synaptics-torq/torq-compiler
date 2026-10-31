"""Average pool operator fixtures built with ONNX GraphSurgeon."""

import ml_dtypes
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def make_localavgpool() -> onnx.ModelProto:
    x = gs.Variable("x", dtype=ml_dtypes.bfloat16, shape=[1, 192, 35, 35])
    output = gs.Variable("output", dtype=ml_dtypes.bfloat16, shape=[1, 192, 35, 35])
    graph = gs.Graph(
        nodes=[
            gs.Node(
                "AveragePool",
                inputs=[x],
                outputs=[output],
                attrs={
                    "ceil_mode": 0,
                    "count_include_pad": 1,
                    "kernel_shape": [3, 3],
                    "pads": [1, 1, 1, 1],
                    "strides": [1, 1],
                },
            ),
        ],
        inputs=[x],
        outputs=[output],
        opset=18,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "localavgpool": make_localavgpool,
}
