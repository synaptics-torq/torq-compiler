"""Keras test models: tanh, sigmoid and ReLU activations."""

from torq.testing.keras_builder import layers, sequential

# int16 tanh/sigmoid: the TOSA table lowering differs from the TFLite int16 kernel by up to 2 LSB
# across the full input range (llvm-cpu matches Torq bit-exactly).
_TABLE_CASES = {
    "model081_tanh_zpm26": sequential((256,), layers.Activation("tanh")),
    "model082_tanh_zpm115": sequential((256,), layers.Activation("tanh")),
    "model083_tanh_zpm128": sequential((256,), layers.Activation("tanh")),
    "model084_tanh_zp76": sequential((1000,), layers.Activation("tanh")),
    "model085_tanh_zp127": sequential((40,), layers.Activation("tanh")),
    "model086_tanh_1x4x6x8": sequential((4, 6, 8), layers.Activation("tanh")),
    "model087_sigmoid_1x4x6x8": sequential((4, 6, 8), layers.Activation("sigmoid")),
    "model088_sigmoid_1x1x16x128": sequential((1, 16, 128), layers.Activation("sigmoid")),
    "model090_sigmoid_zp26": sequential((1, 1, 256), layers.Activation("sigmoid")),
    "model091_sigmoid_zp128": sequential((512,), layers.Activation("sigmoid")),
    "model092_sigmoid_zpm76": sequential((1, 1, 100), layers.Activation("sigmoid")),
    "model093_sigmoid_zp128": sequential((1, 1, 256), layers.Activation("sigmoid")),
    "model094_sigmoid_zp16": sequential((1, 1, 256), layers.Activation("sigmoid")),
}

CASES = {
    **_TABLE_CASES,
    "model095_relu_8x8_inp_1x10x1x1": sequential((10, 1, 1), layers.ReLU()),
    "model096_relu_zp26_8x8_inp_1x10x1x1": sequential((10, 1, 1), layers.ReLU()),
}

CONFIG = {stem: {"int16": {"comparison": {"int_tol": 2}}} for stem in _TABLE_CASES}
