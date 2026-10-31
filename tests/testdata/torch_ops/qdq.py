"""Qdq operator fixtures built with torq.testing.onnx_builder."""

import numpy as np

from torq.testing.onnx_builder import F32, I8, OnnxGraph

KIND = "onnx"


# int8 quantization parameters (scale, zero point) of the MobileNet QDQ mul.
_X = (0.00784313678741455, -1)
_Y = (0.007843135856091976, 0)
_OUT = (0.007838194258511066, -1)


def _qdq_mul(g, a, b, shape, *, annotate=False):
    """Quantized Mul of two int8 operands, each given as (tensor, (scale, zero point)).

    annotate declares the f32 types of the intermediates.
    """
    def types(x_shape):
        return {"dtype": F32, "shape": x_shape} if annotate else {}

    (a, a_q), (b, b_q) = a, b
    product = g.op("Mul", g.dequantize(a, *a_q, **types(a.shape)),
                   g.dequantize(b, *b_q, **types(b.shape)), **types(shape))
    return g.export(g.quantize(product, *_OUT, dtype=I8, shape=shape))


def make_qdq_mul():
    g = OnnxGraph(opset=17)
    shape = [1, 32, 112, 112]
    x = g.input("input_0", I8, shape)
    y = g.input("input_1", I8, shape)
    return _qdq_mul(g, (y, _Y), (x, _X), shape)


def make_qdq_mul_broadcast_const_scale_1ch_remainder():
    rng = np.random.default_rng(0)
    g = OnnxGraph(opset=17)
    shape = [1, 9, 112, 112]
    x = g.input("x", I8, shape)
    y = g.const(rng.integers(-50, 7, [1, 9, 1, 1], dtype=np.int8), I8)
    return _qdq_mul(g, (x, _X), (y, _Y), shape)


def make_qdq_mul_scalar_const():
    g = OnnxGraph(opset=17)
    shape = [1, 32, 112, 112]
    x = g.input("x", I8, shape)
    return _qdq_mul(g, (x, _X), (g.const(37, I8), _Y), shape)


def make_qdq_sigmoid():
    g = OnnxGraph(opset=17)
    x = g.input("x", I8, [1, 32, 112, 112])
    sigmoid = g.op("Sigmoid", g.dequantize(x, 0.007843135856091976, 0))
    return g.export(g.quantize(sigmoid, 0.0028668963350355625, -128, dtype=I8,
                               shape=[1, 32, 112, 112]))


def make_qdq_min_i8():
    g = OnnxGraph(opset=19)
    x = g.input("x", F32, [1, 64, 64])
    return g.export(g.qdq(x, 0.007843137718737125, -7, dtype=F32, shape=[1, 64, 64]))


def make_qdq_mul_broadcast_const_scale():
    rng = np.random.default_rng(0)
    g = OnnxGraph(opset=17)
    shape = [1, 32, 112, 112]
    x = g.input("x", I8, shape)
    y = g.const(rng.integers(-50, 49, [1, 32, 1, 1], dtype=np.int8), I8)
    return _qdq_mul(g, (x, _X), (y, _Y), shape, annotate=True)


def make_qdq_mul_broadcast():
    g = OnnxGraph(opset=17)
    shape = [1, 32, 112, 112]
    x = g.input("x", I8, shape)
    y = g.input("y", I8, [1, 32, 1, 1])
    return _qdq_mul(g, (x, _X), (y, _Y), shape, annotate=True)


CASES = {
    "qdq-mul-broadcast-const-scale": make_qdq_mul_broadcast_const_scale,
    "qdq-mul-broadcast": make_qdq_mul_broadcast,
    "qdq-mul": make_qdq_mul,
    "qdq-mul-broadcast-const-scale-1ch-remainder": make_qdq_mul_broadcast_const_scale_1ch_remainder,
    "qdq-mul-scalar-const": make_qdq_mul_scalar_const,
    "qdq-sigmoid": make_qdq_sigmoid,
    "qdq_min_i8": make_qdq_min_i8,
}

CONFIG = {
    # int8 QDQ zero-point regression: the residual is bf16 MulOp rounding.
    "qdq_min_i8": {"comparison": {"fp_max_tol": 0.2, "allowed_wrong": 0.06}},
}
