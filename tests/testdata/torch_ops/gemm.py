"""Gemm operator fixtures built with ONNX GraphSurgeon."""

from functools import partial

import ml_dtypes
import numpy as np
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def _make_gemm(input_shape, weight_shape, *, weight_range=None, bias_range=None) -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    x = gs.Variable("x", dtype=ml_dtypes.bfloat16, shape=input_shape)
    output = gs.Variable(
        "output", dtype=ml_dtypes.bfloat16, shape=[input_shape[0], weight_shape[1]]
    )
    weights = gs.Constant(
        "weights",
        (
            np.full(weight_shape, 0.010009765625, dtype=ml_dtypes.bfloat16)
            if weight_range is None
            else rng.uniform(*weight_range, weight_shape).astype(ml_dtypes.bfloat16)
        ),
    )
    bias_shape = [weight_shape[1]]
    bias = gs.Constant(
        "bias",
        (
            np.full(bias_shape, 0.0, dtype=ml_dtypes.bfloat16)
            if bias_range is None
            else rng.uniform(*bias_range, bias_shape).astype(ml_dtypes.bfloat16)
        ),
    )
    graph = gs.Graph(
        nodes=[
            gs.Node(
                "Gemm",
                inputs=[x, weights, bias],
                outputs=[output],
                attrs={"alpha": 1.0, "beta": 1.0, "transA": 0, "transB": 0},
            )
        ],
        inputs=[x],
        outputs=[output],
        opset=22,
    )
    return gs.export_onnx(graph, ir_version=10)


def make_qdq_gemm() -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    x = gs.Variable('x', dtype=np.int8, shape=[1, 512])
    output = gs.Variable('output', dtype=np.int8, shape=[1, 1000])
    zero_point = gs.Constant('zero_point', np.full([], -1, dtype=np.int8))
    scale = gs.Constant('scale', np.full([], 0.00784226506948471, dtype=np.float32))
    zero_point_2 = gs.Constant('zero_point_2', np.full([], 44, dtype=np.int8))
    scale_2 = gs.Constant('scale_2', np.full([], 0.11942443996667862, dtype=np.float32))
    zero_point_3 = gs.Constant('zero_point_3', np.full([], 0, dtype=np.int8))
    scale_3 = gs.Constant('scale_3', np.full([], 0.028673745691776276, dtype=np.float32))
    quantized = gs.Constant('quantized', rng.integers(-98, 128, [1000, 512], dtype=np.int8))
    quantized_2 = gs.Constant('quantized_2', rng.integers(-20485, -11013, [1000], dtype=np.int32))
    scale_4 = gs.Constant('scale_4', np.full([1], 0.00022486710804514587, dtype=np.float32))
    zero_point_4 = gs.Constant('zero_point_4', np.full([], 0, dtype=np.int32))
    dequantizelinear = gs.Variable('dequantizelinear')
    dequantizelinear_2 = gs.Variable('dequantizelinear_2')
    dequantizelinear_3 = gs.Variable('dequantizelinear_3')
    gemm = gs.Variable('gemm')
    graph = gs.Graph(
        nodes=[
            gs.Node('DequantizeLinear', inputs=[quantized_2, scale_4, zero_point_4], outputs=[dequantizelinear]),
            gs.Node('DequantizeLinear', inputs=[quantized, scale_3, zero_point_3], outputs=[dequantizelinear_2]),
            gs.Node('DequantizeLinear', inputs=[x, scale, zero_point], outputs=[dequantizelinear_3]),
            gs.Node('Gemm', inputs=[dequantizelinear_3, dequantizelinear_2, dequantizelinear], outputs=[gemm], attrs={'alpha': 1.0, 'beta': 1.0, 'transB': 1}),
            gs.Node('QuantizeLinear', inputs=[gemm, scale_2, zero_point_2], outputs=[output]),
        ],
        inputs=[x],
        outputs=[output],
        opset=18,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "gemm-fc-row-channel-tiling-bf16": partial(_make_gemm, [207, 1152], [1152, 288]),
    "gemm_bf16": partial(
        _make_gemm,
        [207, 1152],
        [1152, 288],
        weight_range=(-3.671875, 3.546875),
        bias_range=(-0.94921875, 0.859375),
    ),
    'qdq-gemm': make_qdq_gemm,
}
