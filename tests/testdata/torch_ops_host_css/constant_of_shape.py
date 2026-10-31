"""Constant of shape operator fixtures built with torq.testing.onnx_builder."""

from torq.testing.onnx_builder import I64, OnnxGraph

KIND = "onnx"


def make_constantshape():
    g = OnnxGraph(opset=18)
    y = g.op("ConstantOfShape", g.const([2], I64), value=g.const([1], I64), dtype=I64, shape=[2])
    return g.export(y)


CASES = {
    "constantshape": make_constantshape,
}
