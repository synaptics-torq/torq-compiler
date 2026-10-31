"""Broadcast operator fixtures built with ONNX GraphSurgeon."""

import ml_dtypes
import numpy as np
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def make_collapse_issue_moonshine() -> onnx.ModelProto:
    input_0 = gs.Variable("input_0", dtype=ml_dtypes.bfloat16, shape=[1, 1, 8, 36])
    input_1 = gs.Variable("input_1", dtype=ml_dtypes.bfloat16, shape=[1, 1, 8, 36])
    input_2 = gs.Variable("input_2", dtype=ml_dtypes.bfloat16, shape=[1, 1, 8, 36])
    input_3 = gs.Variable("input_3", dtype=ml_dtypes.bfloat16, shape=[1, 1, 1, 32])
    input_4 = gs.Variable("input_4", dtype=ml_dtypes.bfloat16, shape=[1, 1, 32])
    output_0 = gs.Variable("output_0", dtype=ml_dtypes.bfloat16, shape=[1, 1, 8, 32])
    output_1 = gs.Variable("output_1", dtype=ml_dtypes.bfloat16, shape=[1, 8, 1, 4])
    output_2 = gs.Variable("output_2", dtype=ml_dtypes.bfloat16, shape=[1, 8, 1, 32])
    output_3 = gs.Variable("output_3", dtype=ml_dtypes.bfloat16, shape=[1, 8, 1, 4])
    output_4 = gs.Variable("output_4", dtype=ml_dtypes.bfloat16, shape=[1, 8, 1, 32])
    output_5 = gs.Variable("output_5", dtype=ml_dtypes.bfloat16, shape=[1, 8, 30, 36])
    output_6 = gs.Variable("output_6", dtype=ml_dtypes.bfloat16, shape=[1, 1, 1, 32])
    output_7 = gs.Variable("output_7", dtype=ml_dtypes.bfloat16, shape=[1, 1, 1, 32])
    feature_axis = gs.Constant("feature_axis", np.full([1], 3, dtype=np.int64))
    tail_start = gs.Constant("tail_start", np.full([1], 32, dtype=np.int64))
    tail_end = gs.Constant("tail_end", np.full([1], 9223372036854775807, dtype=np.int64))
    broadcast_shape = gs.Constant("broadcast_shape", np.array([1, 8, 30, 36], dtype=np.int64))
    unit_axis = gs.Constant("unit_axis", np.full([1], 1, dtype=np.int64))
    prefix_start = gs.Constant("prefix_start", np.full([1], 0, dtype=np.int64))
    tail = gs.Variable("tail")
    prefix = gs.Variable("prefix")
    transposed_1 = gs.Variable("transposed_1")
    transposed_0 = gs.Variable("transposed_0")
    graph = gs.Graph(
        nodes=[
            gs.Node("Unsqueeze", inputs=[input_4, unit_axis], outputs=[output_6]),
            gs.Node(
                "Slice",
                inputs=[input_2, tail_start, tail_end, feature_axis, unit_axis],
                outputs=[tail],
            ),
            gs.Node(
                "Slice",
                inputs=[input_2, prefix_start, tail_start, feature_axis, unit_axis],
                outputs=[prefix],
            ),
            gs.Node(
                "Transpose", inputs=[input_1], outputs=[transposed_1], attrs={"perm": [0, 2, 1, 3]}
            ),
            gs.Node(
                "Transpose", inputs=[input_0], outputs=[transposed_0], attrs={"perm": [0, 2, 1, 3]}
            ),
            gs.Node(
                "Transpose", inputs=[input_3], outputs=[output_7], attrs={"perm": [0, 2, 1, 3]}
            ),
            gs.Node("Transpose", inputs=[tail], outputs=[output_3], attrs={"perm": [0, 2, 1, 3]}),
            gs.Node("Transpose", inputs=[prefix], outputs=[output_2], attrs={"perm": [0, 2, 1, 3]}),
            gs.Node(
                "Slice",
                inputs=[transposed_1, prefix_start, tail_start, feature_axis, unit_axis],
                outputs=[output_4],
            ),
            gs.Node(
                "Slice",
                inputs=[transposed_1, tail_start, tail_end, feature_axis, unit_axis],
                outputs=[output_1],
            ),
            gs.Node("Expand", inputs=[transposed_0, broadcast_shape], outputs=[output_5]),
            gs.Node("Mul", inputs=[prefix, output_7], outputs=[output_0]),
        ],
        inputs=[input_0, input_1, input_2, input_3, input_4],
        outputs=[output_0, output_1, output_2, output_3, output_4, output_5, output_6, output_7],
        opset=22,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "collapse-issue-moonshine": make_collapse_issue_moonshine,
}
