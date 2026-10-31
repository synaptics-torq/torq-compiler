"""Global average pool operator fixtures built with ONNX GraphSurgeon."""

import ml_dtypes
import numpy as np
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def make_GlobalAveragePool() -> onnx.ModelProto:
    x = gs.Variable("x", dtype=ml_dtypes.bfloat16, shape=[1, 2560, 8, 8])
    output = gs.Variable("output", dtype=ml_dtypes.bfloat16, shape=[1, 2560, 1, 1])
    graph = gs.Graph(
        nodes=[
            gs.Node("GlobalAveragePool", inputs=[x], outputs=[output]),
        ],
        inputs=[x],
        outputs=[output],
        opset=18,
    )
    return gs.export_onnx(graph, ir_version=10)


def make_quant_globalAvgPool() -> onnx.ModelProto:
    x = gs.Variable("x", dtype=np.int8, shape=[1, 512, 7, 7])
    output = gs.Variable("output", dtype=np.int8, shape=[1, 512, 1, 1])
    zero_point = gs.Constant("zero_point", np.full([], 0, dtype=np.int8))
    scale = gs.Constant("scale", np.full([], 0.007843106053769588, dtype=np.float32))
    zero_point_2 = gs.Constant("zero_point_2", np.full([], -5, dtype=np.int8))
    scale_2 = gs.Constant("scale_2", np.full([], 0.002344646258279681, dtype=np.float32))
    dequantizelinear = gs.Variable("dequantizelinear")
    globalaveragepool = gs.Variable("globalaveragepool")
    graph = gs.Graph(
        nodes=[
            gs.Node("DequantizeLinear", inputs=[x, scale, zero_point], outputs=[dequantizelinear]),
            gs.Node("GlobalAveragePool", inputs=[dequantizelinear], outputs=[globalaveragepool]),
            gs.Node(
                "QuantizeLinear",
                inputs=[globalaveragepool, scale_2, zero_point_2],
                outputs=[output],
            ),
        ],
        inputs=[x],
        outputs=[output],
        opset=18,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "GlobalAveragePool": make_GlobalAveragePool,
    "quant-globalAvgPool": make_quant_globalAvgPool,
}
