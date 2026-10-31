"""Resize operator fixtures built with ONNX GraphSurgeon."""

import ml_dtypes
import numpy as np
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def make_resize_nearest_upsample_fpn_bf16() -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    input_0 = gs.Variable("input_0", dtype=ml_dtypes.bfloat16, shape=[1, 16, 10, 10])
    input_1 = gs.Variable("input_1", dtype=ml_dtypes.bfloat16, shape=[1, 16, 20, 20])
    input_2 = gs.Variable("input_2", dtype=ml_dtypes.bfloat16, shape=[1, 16, 40, 40])
    input_3 = gs.Variable("input_3", dtype=ml_dtypes.bfloat16, shape=[1, 16, 80, 80])
    output = gs.Variable("output", dtype=ml_dtypes.bfloat16, shape=[1, 16, 80, 80])
    roi = gs.Constant("roi", np.empty([0], dtype=np.float32))
    scales = gs.Constant("scales", np.array([1.0, 1.0, 2.0, 2.0], dtype=np.float32))
    scales_2 = gs.Constant("scales_2", np.array([1.0, 1.0, 8.0, 8.0], dtype=np.float32))
    scales_3 = gs.Constant("scales_3", np.array([1.0, 1.0, 4.0, 4.0], dtype=np.float32))
    weights = gs.Constant(
        "weights",
        rng.uniform(-0.00640869140625, 0.00592041015625, [16, 64, 3, 3]).astype(ml_dtypes.bfloat16),
    )
    bias = gs.Constant(
        "bias", rng.uniform(-0.337890625, 0.96484375, [16]).astype(ml_dtypes.bfloat16)
    )
    resize = gs.Variable("resize")
    resize_2 = gs.Variable("resize_2")
    resize_3 = gs.Variable("resize_3")
    concat = gs.Variable("concat")
    graph = gs.Graph(
        nodes=[
            gs.Node(
                "Resize",
                inputs=[input_0, roi, scales_2],
                outputs=[resize],
                attrs={
                    "coordinate_transformation_mode": "asymmetric",
                    "mode": "nearest",
                    "nearest_mode": "floor",
                },
            ),
            gs.Node(
                "Resize",
                inputs=[input_1, roi, scales_3],
                outputs=[resize_2],
                attrs={
                    "coordinate_transformation_mode": "asymmetric",
                    "mode": "nearest",
                    "nearest_mode": "floor",
                },
            ),
            gs.Node(
                "Resize",
                inputs=[input_2, roi, scales],
                outputs=[resize_3],
                attrs={
                    "coordinate_transformation_mode": "asymmetric",
                    "mode": "nearest",
                    "nearest_mode": "floor",
                },
            ),
            gs.Node(
                "Concat",
                inputs=[resize, resize_2, resize_3, input_3],
                outputs=[concat],
                attrs={"axis": 1},
            ),
            gs.Node(
                "Conv",
                inputs=[concat, weights, bias],
                outputs=[output],
                attrs={
                    "dilations": [1, 1],
                    "group": 1,
                    "kernel_shape": [3, 3],
                    "pads": [1, 1, 1, 1],
                    "strides": [1, 1],
                },
            ),
        ],
        inputs=[input_0, input_1, input_2, input_3],
        outputs=[output],
        opset=22,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "resize_nearest_upsample_fpn_bf16": make_resize_nearest_upsample_fpn_bf16,
}
