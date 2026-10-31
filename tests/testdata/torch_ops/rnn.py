"""Rnn operator fixtures built with ONNX GraphSurgeon."""

import ml_dtypes
import numpy as np
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def make_rnn_s4_b1_h8_bf16() -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    input_0 = gs.Variable("input_0", dtype=ml_dtypes.bfloat16, shape=[4, 1, 8])
    input_1 = gs.Variable("input_1", dtype=ml_dtypes.bfloat16, shape=[1, 1, 8])
    output_0 = gs.Variable("output_0", dtype=ml_dtypes.bfloat16, shape=[4, 1, 1, 8])
    output_1 = gs.Variable("output_1", dtype=ml_dtypes.bfloat16, shape=[1, 1, 8])
    rnn_constant = gs.Constant(
        "rnn_constant", rng.uniform(-0.337890625, 0.3515625, [1, 8, 8]).astype(ml_dtypes.bfloat16)
    )
    rnn_constant_2 = gs.Constant(
        "rnn_constant_2",
        rng.uniform(-0.353515625, 0.33984375, [1, 8, 8]).astype(ml_dtypes.bfloat16),
    )
    rnn_constant_3 = gs.Constant(
        "rnn_constant_3", rng.uniform(-0.2890625, 0.3359375, [1, 16]).astype(ml_dtypes.bfloat16)
    )
    graph = gs.Graph(
        nodes=[
            gs.Node(
                "RNN",
                inputs=[
                    input_0,
                    rnn_constant,
                    rnn_constant_2,
                    rnn_constant_3,
                    gs.Variable.empty(),
                    input_1,
                ],
                outputs=[output_0, output_1],
                attrs={"activations": ["Tanh"], "hidden_size": 8},
            ),
        ],
        inputs=[input_0, input_1],
        outputs=[output_0, output_1],
        opset=17,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "rnn_s4_b1_h8_bf16": make_rnn_s4_b1_h8_bf16,
}
