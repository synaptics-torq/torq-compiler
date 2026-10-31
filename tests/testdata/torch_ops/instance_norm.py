"""ONNX operator fixtures built with GraphSurgeon."""

from torq.testing.onnx_builder import BF16, OnnxGraph

KIND = "onnx"


def make_instancenorm():
    g = OnnxGraph(opset=22)
    shape = [1, 1, 359712]
    x = g.input("x", BF16, shape)
    scale, bias = g.const([1.0], BF16), g.const([0.0], BF16)
    y = g.op("InstanceNormalization", x, scale, bias, epsilon=9.99999974e-06, dtype=BF16,
             shape=shape)
    return g.export(y)


CASES = {
    "instancenorm": make_instancenorm,
}

CONFIG = {
    "instancenorm": {
        # torch-mlir 3.12 decomposes InstanceNormalization in f32 inside the dispatch (bf16
        # interfaces); the NSS elementwise path needs bf16, so demote the cone back.
        "compiler_options": ["--torq-convert-dtypes"],
        # The bf16 output carries ~1 bf16 ULP of error: against the fp32 numpy reference a few
        # near-zero elements just exceed the default fp_max_tol.
        "comparison": {"fp_max_tol": 2e-2, "allowed_wrong": 1e-3},
        "reference": "numpy_instancenorm_reference_results",
    },
}
