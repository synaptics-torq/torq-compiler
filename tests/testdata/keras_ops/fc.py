"""Keras test models: Dense layers (fully connected), with activations.

Each case name says what it tests: input size, output size and the activation after the layer,
e.g. fc_in61_out1991_relu.
"""

from torq.testing.keras_builder import layers, sequential

_ACTIVATIONS = {
    "relu": lambda: layers.ReLU(),
    "relu6": lambda: layers.ReLU(6.0),
    "sigmoid": lambda: layers.Activation("sigmoid"),
    "softmax": lambda: layers.Softmax(),
}

# input size, output size, activation (None: none)
_LAYERS = [
    (1, 1, "relu6"),
    (61, 1991, "relu"),
    (1024, 1024, None),
    (1000, 512, "relu6"),
    (2000, 97, "relu6"),
    (700, 500, "sigmoid"),
    (1000, 1000, "softmax"),
    (3, 7, "relu"),
]


def _name(inputs, units, act):
    return f"fc_in{inputs}_out{units}" + (f"_{act}" if act else "")


CASES = {
    # three layers: 1 -> 16 -> 16 -> 1
    "fc_in1_out16_relu_out16_relu_out1": sequential(
        (1,),
        layers.Dense(units=16),
        layers.ReLU(),
        layers.Dense(units=16),
        layers.ReLU(),
        layers.Dense(units=1),
    ),
}
for _inputs, _units, _act in _LAYERS:
    _key = _name(_inputs, _units, _act)
    if _key in CASES:
        raise ValueError(f"fc: {_key} is listed twice")
    CASES[_key] = sequential((_inputs,), layers.Dense(units=_units),
                             *([_ACTIVATIONS[_act]()] if _act else []))
