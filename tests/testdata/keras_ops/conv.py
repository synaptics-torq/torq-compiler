"""Keras test models: Conv2D, alone or followed by ReLU or another conv.

Each case name says what it tests: kernel, strides, padding, input shape (H x W x C), output
channels and the activation after the conv, e.g. conv2d_k3x3_s2x2_same_24x32x1_oc4_relu6.
"""

from functools import partial

from tests.models.keras_models import build_conv_model, conv_model_params
from torq.testing.keras_builder import layers, sequential, conv2d

_ACTIVATIONS = {"relu": lambda: layers.ReLU(), "relu6": lambda: layers.ReLU(6.0)}

# (input H, W, C), output channels, kernel (h, w), strides (h, w), padding
_CONVS = [
    ((3, 4, 2), 3, (2, 3), (1, 1), "valid"),
    ((7, 30, 19), 4, (4, 5), (1, 1), "valid"),
    ((38, 50, 12), 20, (4, 1), (1, 1), "valid"),
    ((39, 26, 14), 20, (1, 2), (1, 1), "same"),
    ((58, 9, 24), 19, (5, 2), (1, 1), "same"),
    ((41, 19, 5), 32, (5, 4), (4, 4), "valid"),
    ((22, 55, 21), 9, (2, 5), (1, 1), "same"),
    ((18, 26, 19), 17, (1, 1), (1, 1), "valid"),
    ((4, 3, 26), 25, (2, 1), (1, 1), "same"),
    ((16, 55, 7), 13, (5, 3), (3, 3), "same"),
    ((20, 42, 9), 25, (1, 5), (1, 1), "same"),
    ((5, 40, 24), 12, (4, 3), (1, 1), "same"),
    ((34, 6, 2), 26, (5, 4), (3, 3), "valid"),
    ((13, 16, 29), 15, (4, 2), (1, 1), "valid"),
    ((60, 12, 9), 22, (1, 3), (1, 1), "valid"),
    ((45, 22, 27), 8, (4, 4), (3, 3), "same"),
    ((54, 63, 7), 16, (5, 2), (1, 1), "valid"),
    ((6, 46, 11), 9, (4, 3), (1, 1), "same"),
    ((15, 13, 10), 17, (3, 5), (3, 3), "valid"),
    ((12, 52, 10), 23, (5, 4), (2, 2), "valid"),
    ((43, 61, 20), 26, (3, 3), (3, 3), "same"),
    ((18, 76, 24), 1, (5, 2), (1, 1), "same"),
    ((58, 26, 24), 1, (5, 2), (1, 1), "same"),
]

# the same, then an activation
_CONVS_ACT = [
    ((24, 32, 1), 4, (3, 3), (2, 2), "same", "relu6"),
    ((3, 4, 2), 3, (3, 3), (1, 1), "valid", "relu"),
    ((3, 4, 3), 4, (3, 3), (1, 1), "same", "relu"),
    ((13, 35, 1), 4, (6, 6), (2, 2), "same", "relu"),
    ((12, 14, 4), 8, (3, 3), (1, 1), "same", "relu"),
    ((12, 14, 4), 4, (4, 4), (1, 1), "same", "relu"),
]

# two "same" convs with 2 output channels on a 4x4x2 input, each followed by ReLU: the kernels
_TWO_CONVS = [((3, 3), (2, 2)), ((3, 3), (3, 3)), ((2, 2), (3, 3))]


def _name(shape, filters, kernel, strides, padding, after=""):
    h, w, c = shape
    return (f"conv2d_k{kernel[0]}x{kernel[1]}_s{strides[0]}x{strides[1]}_{padding}_"
            f"{h}x{w}x{c}_oc{filters}{after}")


def _conv_act(shape, filters, kernel, strides, padding, act):
    return sequential(shape, layers.Conv2D(filters=filters, kernel_size=kernel, strides=strides,
                                           padding=padding), _ACTIVATIONS[act]())


def _two_convs(k1, k2):
    return sequential(
        (4, 4, 2),
        layers.Conv2D(filters=2, kernel_size=k1, padding="same"), layers.ReLU(),
        layers.Conv2D(filters=2, kernel_size=k2, padding="same"), layers.ReLU(),
    )


CASES = {}


def _add(name, builder):
    if name in CASES:
        raise ValueError(f"conv: {name} is listed twice")
    CASES[name] = builder


for _c in _CONVS:
    _add(_name(*_c), conv2d(*_c))
for *_c, _act in _CONVS_ACT:
    _add(_name(*_c, f"_{_act}"), _conv_act(*_c, _act))
for _k1, _k2 in _TWO_CONVS:
    _add(_name((4, 4, 2), 2, _k1, (1, 1), "same", f"_relu_then_k{_k2[0]}x{_k2[1]}_relu"),
         _two_convs(_k1, _k2))
# The conv_model of test_conv_sweep, with the two default parameter sets.
for _p in conv_model_params:
    _add(f"conv_model_{_p.idfn()}", partial(build_conv_model, _p))

# A string quantize gives a single case with no _int8/_int16 suffix.
CONFIG = {f"conv_model_{p.idfn()}": {"quantize": "int8"} for p in conv_model_params}
