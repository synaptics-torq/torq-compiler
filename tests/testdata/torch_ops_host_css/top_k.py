"""Top k operator fixtures built with ONNX GraphSurgeon."""

import ml_dtypes
import numpy as np
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def make_topk_1x2100_bf16() -> onnx.ModelProto:
    x = gs.Variable("x", dtype=ml_dtypes.bfloat16, shape=[1, 2100])
    output_0 = gs.Variable("output_0", dtype=ml_dtypes.bfloat16, shape=[1, 300])
    output_1 = gs.Variable("output_1", dtype=np.int64, shape=[1, 300])
    k = gs.Constant("k", np.full([1], 300, dtype=np.int64))
    graph = gs.Graph(
        nodes=[
            gs.Node(
                "TopK",
                inputs=[x, k],
                outputs=[output_0, output_1],
                attrs={"axis": -1, "largest": 1, "sorted": 1},
            ),
        ],
        inputs=[x],
        outputs=[output_0, output_1],
        opset=22,
    )
    return gs.export_onnx(graph, ir_version=10)


def make_topk_1x2100_indices_only_bf16() -> onnx.ModelProto:
    x = gs.Variable("x", dtype=ml_dtypes.bfloat16, shape=[1, 2100])
    output = gs.Variable("output", dtype=np.int64, shape=[1, 300])
    k = gs.Constant("k", np.full([1], 300, dtype=np.int64))
    topk_0 = gs.Variable("topk_0")
    graph = gs.Graph(
        nodes=[
            gs.Node(
                "TopK",
                inputs=[x, k],
                outputs=[topk_0, output],
                attrs={"axis": -1, "largest": 1, "sorted": 1},
            ),
        ],
        inputs=[x],
        outputs=[output],
        opset=22,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "topk-1x2100-bf16": make_topk_1x2100_bf16,
    "topk-1x2100-indices-only-bf16": make_topk_1x2100_indices_only_bf16,
}
