"""Layer norm operator fixtures built with ONNX GraphSurgeon."""

from functools import partial

import ml_dtypes
import numpy as np
import onnx
import onnx_graphsurgeon as gs

KIND = "onnx"


def _make_layer_norm(
    input_shape, *, scale_low=0.5, scale_high=1.5, bias_low=-1.0, bias_high=1.0
) -> onnx.ModelProto:
    rng = np.random.default_rng(0)
    x = gs.Variable("x", dtype=ml_dtypes.bfloat16, shape=input_shape)
    output = gs.Variable("output", dtype=ml_dtypes.bfloat16, shape=input_shape)
    scale = gs.Constant(
        "scale", rng.uniform(scale_low, scale_high, [input_shape[-1]]).astype(ml_dtypes.bfloat16)
    )
    bias = gs.Constant(
        "bias", rng.uniform(bias_low, bias_high, [input_shape[-1]]).astype(ml_dtypes.bfloat16)
    )
    graph = gs.Graph(
        nodes=[
            gs.Node(
                "LayerNormalization",
                inputs=[x, scale, bias],
                outputs=[output],
                attrs={"axis": -1, "epsilon": 9.999999747378752e-06, "stash_type": 16},
            )
        ],
        inputs=[x],
        outputs=[output],
        opset=18,
    )
    return gs.export_onnx(graph, ir_version=10)


CASES = {
    "layernorm-nss-1x128x768-bf16": partial(_make_layer_norm, [1, 128, 768]),
    "layernorm_8x1x32": partial(
        _make_layer_norm,
        [8, 1, 32],
        scale_low=0.73046875,
        scale_high=1.4296875,
        bias_low=-3.421875,
        bias_high=3.265625,
    ),
}
