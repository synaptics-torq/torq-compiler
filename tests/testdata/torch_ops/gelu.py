"""Gelu operator fixtures built with ONNX GraphSurgeon."""

import ml_dtypes
import numpy as np
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def make_gelu_gemma() -> onnx.ModelProto:
    x = gs.Variable("x", dtype=ml_dtypes.bfloat16, shape=[1, 1024, 64])
    output = gs.Variable("output", dtype=ml_dtypes.bfloat16, shape=[1, 1024, 64])
    graph = gs.Graph(
        nodes=[
            gs.Node("Gelu", inputs=[x], outputs=[output], attrs={"approximate": "tanh"}),
        ],
        inputs=[x],
        outputs=[output],
        opset=22,
    )
    return gs.export_onnx(graph, ir_version=10)


def make_erf_mlp_dup_lram_load_bf16() -> onnx.ModelProto:
    """PP-OCR MLP regression for duplicate GELU operand loads in LRAM.

    At [1,192,3,608], two identical loads need 705,544 B; merging them needs
    472,072 B and fits the 523,008 B LRAM. Equal-index frees must precede
    allocations in the interval sweep or the merge can be rejected.
    Preserve the ONNX Div/Erf/Add/Mul/Mul layout and bf16-exact splat weights:
    torch.export changes the layout and does not reproduce this failure.
    """
    x = gs.Variable("x", dtype=ml_dtypes.bfloat16, shape=[1, 96, 3, 608])
    convolved = gs.Variable("convolved", dtype=ml_dtypes.bfloat16, shape=[1, 192, 3, 608])
    divided = gs.Variable("divided", dtype=ml_dtypes.bfloat16, shape=[1, 192, 3, 608])
    erf = gs.Variable("erf", dtype=ml_dtypes.bfloat16, shape=[1, 192, 3, 608])
    shifted = gs.Variable("shifted", dtype=ml_dtypes.bfloat16, shape=[1, 192, 3, 608])
    product = gs.Variable("product", dtype=ml_dtypes.bfloat16, shape=[1, 192, 3, 608])
    gelu = gs.Variable("gelu", dtype=ml_dtypes.bfloat16, shape=[1, 192, 3, 608])
    output = gs.Variable("output", dtype=ml_dtypes.bfloat16, shape=[1, 160, 3, 608])
    sqrt_two = gs.Constant("sqrt_two", np.full([], 1.4140625, dtype=ml_dtypes.bfloat16))
    one = gs.Constant("one", np.full([], 1.0, dtype=ml_dtypes.bfloat16))
    half = gs.Constant("half", np.full([], 0.5, dtype=ml_dtypes.bfloat16))
    weights1 = gs.Constant("weights1", np.full([192, 96, 1, 1], 0.0625, dtype=ml_dtypes.bfloat16))
    bias1 = gs.Constant("bias1", np.full([192], 0.125, dtype=ml_dtypes.bfloat16))
    weights2 = gs.Constant("weights2", np.full([160, 192, 1, 1], -0.03125, dtype=ml_dtypes.bfloat16))
    bias2 = gs.Constant("bias2", np.full([160], 0.09375, dtype=ml_dtypes.bfloat16))
    graph = gs.Graph(
        nodes=[
            gs.Node(
                "Conv",
                inputs=[x, weights1, bias1],
                outputs=[convolved],
                attrs={
                    "dilations": [1, 1],
                    "group": 1,
                    "kernel_shape": [1, 1],
                    "pads": [0, 0, 0, 0],
                    "strides": [1, 1],
                },
            ),
            gs.Node("Div", inputs=[convolved, sqrt_two], outputs=[divided]),
            gs.Node("Erf", inputs=[divided], outputs=[erf]),
            gs.Node("Add", inputs=[erf, one], outputs=[shifted]),
            gs.Node("Mul", inputs=[convolved, shifted], outputs=[product]),
            gs.Node("Mul", inputs=[product, half], outputs=[gelu]),
            gs.Node(
                "Conv",
                inputs=[gelu, weights2, bias2],
                outputs=[output],
                attrs={
                    "dilations": [1, 1],
                    "group": 1,
                    "kernel_shape": [1, 1],
                    "pads": [0, 0, 0, 0],
                    "strides": [1, 1],
                },
            ),
        ],
        inputs=[x],
        outputs=[output],
        opset=22,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "erf_mlp_dup_lram_load_bf16": make_erf_mlp_dup_lram_load_bf16,
    "gelu-gemma": make_gelu_gemma,
}
