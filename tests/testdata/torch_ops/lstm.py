"""Lstm operator fixtures built with ONNX GraphSurgeon."""

import ml_dtypes
import numpy as np
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def make_lstm_s4_b1_h8_bf16() -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    input_0 = gs.Variable("input_0", dtype=ml_dtypes.bfloat16, shape=[4, 1, 8])
    input_1 = gs.Variable("input_1", dtype=ml_dtypes.bfloat16, shape=[1, 1, 8])
    input_2 = gs.Variable("input_2", dtype=ml_dtypes.bfloat16, shape=[1, 1, 8])
    output_0 = gs.Variable("output_0", dtype=ml_dtypes.bfloat16, shape=[4, 1, 1, 8])
    output_1 = gs.Variable("output_1", dtype=ml_dtypes.bfloat16, shape=[1, 1, 8])
    output_2 = gs.Variable("output_2", dtype=ml_dtypes.bfloat16, shape=[1, 1, 8])
    lstm_constant = gs.Constant(
        "lstm_constant", rng.uniform(-0.353515625, 0.3515625, [1, 32, 8]).astype(ml_dtypes.bfloat16)
    )
    lstm_constant_2 = gs.Constant(
        "lstm_constant_2",
        rng.uniform(-0.349609375, 0.353515625, [1, 32, 8]).astype(ml_dtypes.bfloat16),
    )
    lstm_constant_3 = gs.Constant(
        "lstm_constant_3", rng.uniform(-0.34375, 0.353515625, [1, 64]).astype(ml_dtypes.bfloat16)
    )
    graph = gs.Graph(
        nodes=[
            gs.Node(
                "LSTM",
                inputs=[
                    input_0,
                    lstm_constant,
                    lstm_constant_2,
                    lstm_constant_3,
                    gs.Variable.empty(),
                    input_1,
                    input_2,
                ],
                outputs=[output_0, output_1, output_2],
                attrs={"hidden_size": 8},
            ),
        ],
        inputs=[input_0, input_1, input_2],
        outputs=[output_0, output_1, output_2],
        opset=17,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "lstm_s4_b1_h8_bf16": make_lstm_s4_b1_h8_bf16,
}
