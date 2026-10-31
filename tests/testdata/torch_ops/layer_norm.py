"""Layer norm operator fixtures built with torq.testing.onnx_builder."""

from functools import partial

import numpy as np

from torq.testing.onnx_builder import BF16, OnnxGraph

KIND = "onnx"


def _make_layer_norm(input_shape, *, scale_low=0.5, scale_high=1.5, bias_low=-1.0, bias_high=1.0):
    rng = np.random.default_rng(0)
    g = OnnxGraph(opset=18)
    x = g.input("x", BF16, input_shape)
    channels = [input_shape[-1]]
    scale = g.const(rng.uniform(scale_low, scale_high, channels), BF16)
    bias = g.const(rng.uniform(bias_low, bias_high, channels), BF16)
    y = g.op("LayerNormalization", x, scale, bias, axis=-1, epsilon=9.999999747378752e-06,
             stash_type=16, dtype=BF16, shape=input_shape)
    return g.export(y)


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

CONFIG = {
    "layernorm-nss-1x128x768-bf16": {
        "comparison": {
            "fp_max_tol": 0.02, "allowed_wrong": 0.0,
            "use_abs_tol_gate": 1.0, "fp_abs_tol_frac": 0.01,
        },
    },
    # Random affine weights expose the same bf16 rounding near cancellation as the larger
    # LayerNorm case (max absolute difference: 0.03125).
    "layernorm_8x1x32": {
        "comparison": {
            "fp_max_tol": 0.02, "allowed_wrong": 0.0,
            "use_abs_tol_gate": 1.0, "fp_abs_tol_frac": 0.01,
        },
    },
}
