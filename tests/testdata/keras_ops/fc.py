"""Keras test models: Dense layers (fully connected), with activations."""

from torq.testing.keras_builder import layers, sequential

CASES = {
    "model003_hello_world": sequential(
        (1,),
        layers.Dense(units=16),
        layers.ReLU(),
        layers.Dense(units=16),
        layers.ReLU(),
        layers.Dense(units=1),
    ),
    "model020_fc_1x1": sequential((1,), layers.Dense(units=1), layers.ReLU(6.0)),
    "model021_fc_1991x61": sequential((61,), layers.Dense(units=1991), layers.ReLU()),
    "model022_fc_1024x1024": sequential((1024,), layers.Dense(units=1024)),
    "model023_fc_512x1000": sequential((1000,), layers.Dense(units=512), layers.ReLU(6.0)),
    "model024_fc_97x2000": sequential((2000,), layers.Dense(units=97), layers.ReLU(6.0)),
    "model026_fc_1991x61": sequential((61,), layers.Dense(units=1991), layers.ReLU()),
    "model027_fc_500x700": sequential(
        (700,),
        layers.Dense(units=500),
        layers.Activation("sigmoid"),
    ),
    "model029_fc_1000x1000": sequential((1000,), layers.Dense(units=1000), layers.Softmax()),
    "model080_fc_7x3": sequential((3,), layers.Dense(units=7), layers.ReLU()),
}
