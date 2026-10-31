"""Broadcast operator fixtures built with torq.testing.onnx_builder."""

from torq.testing.onnx_builder import BF16, I64, OnnxGraph

KIND = "onnx"


def make_collapse_issue_moonshine():
    g = OnnxGraph(opset=22)
    x0 = g.input("input_0", BF16, [1, 1, 8, 36])
    x1 = g.input("input_1", BF16, [1, 1, 8, 36])
    x2 = g.input("input_2", BF16, [1, 1, 8, 36])
    x3 = g.input("input_3", BF16, [1, 1, 1, 32])
    x4 = g.input("input_4", BF16, [1, 1, 32])
    feature_axis, unit_axis = g.const([3], I64), g.const([1], I64)
    prefix_start, tail_start = g.const([0], I64), g.const([32], I64)
    tail_end = g.const([9223372036854775807], I64)

    def prefix(x, **kwargs):
        return g.op("Slice", x, prefix_start, tail_start, feature_axis, unit_axis, **kwargs)

    def tail(x, **kwargs):
        return g.op("Slice", x, tail_start, tail_end, feature_axis, unit_axis, **kwargs)

    def transpose(x, **kwargs):
        return g.op("Transpose", x, perm=[0, 2, 1, 3], **kwargs)

    t0, t1 = transpose(x0), transpose(x1)
    t3 = transpose(x3, dtype=BF16, shape=[1, 1, 1, 32])
    p2 = prefix(x2)
    return g.export(
        g.op("Mul", p2, t3, dtype=BF16, shape=[1, 1, 8, 32]),
        tail(t1, dtype=BF16, shape=[1, 8, 1, 4]),
        transpose(p2, dtype=BF16, shape=[1, 8, 1, 32]),
        transpose(tail(x2), dtype=BF16, shape=[1, 8, 1, 4]),
        prefix(t1, dtype=BF16, shape=[1, 8, 1, 32]),
        g.op("Expand", t0, g.const([1, 8, 30, 36], I64), dtype=BF16, shape=[1, 8, 30, 36]),
        g.op("Unsqueeze", x4, unit_axis, dtype=BF16, shape=[1, 1, 1, 32]),
        t3,
    )


CASES = {
    "collapse-issue-moonshine": make_collapse_issue_moonshine,
}
