"""Top k operator fixtures built with torq.testing.onnx_builder."""

from torq.testing.onnx_builder import BF16, I64, OnnxGraph, typed

KIND = "onnx"


def _make_topk(g):
    x = g.input("x", BF16, [1, 2100])
    return g.op("TopK", x, g.const([300], I64), outputs=2, axis=-1, largest=1, sorted=1)


def make_topk_1x2100_bf16():
    g = OnnxGraph(opset=22)
    values, indices = _make_topk(g)
    return g.export(typed(values, BF16, [1, 300]), typed(indices, I64, [1, 300]))


def make_topk_1x2100_indices_only_bf16():
    g = OnnxGraph(opset=22)
    _, indices = _make_topk(g)
    return g.export(typed(indices, I64, [1, 300]))


CASES = {
    "topk-1x2100-bf16": make_topk_1x2100_bf16,
    "topk-1x2100-indices-only-bf16": make_topk_1x2100_indices_only_bf16,
}

# Without slicing: the sliced path asserts in LData (Kernel.cpp, "Invalid element type").
# The sort runs as a Host/CSS software kernel, which is slow to simulate over 2100 elements on
# the C-model: it can exceed the 240 s default runtime timeout on a slow CI runner. The extra
# budget applies to the simulator only, so the default still catches a slow sort on FPGA and
# hardware.
CONFIG = {
    "topk-1x2100-bf16": {
        "compiler_options": ["--torq-disable-slicing"],
        "sim_runtime_timeout": 600,
    },
    "topk-1x2100-indices-only-bf16": {
        "compiler_options": ["--torq-disable-slicing"],
        "sim_runtime_timeout": 600,
    },
}
