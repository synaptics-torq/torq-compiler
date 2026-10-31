"""Keras test models: DepthwiseConv2D followed by ReLU, and a global depthwise conv."""

from torq.testing.keras_builder import layers, sequential, depthwise

CASES = {
    "model010_depthwise_1x3x3x16_valid": sequential(
        (6, 42, 16),
        layers.DepthwiseConv2D(kernel_size=(3, 3)),
        layers.ReLU(),
    ),
    "model011_depthwise_1x3x3x16_valid": sequential(
        (6, 4, 16),
        layers.DepthwiseConv2D(kernel_size=(3, 3)),
        layers.ReLU(),
    ),
    "model012_depthwise_1x3x3x2_valid": sequential(
        (6, 6, 2),
        layers.DepthwiseConv2D(kernel_size=(3, 3)),
        layers.ReLU(),
    ),
    "model013_depthwise_1x3x3x2_same": sequential(
        (6, 6, 2),
        layers.DepthwiseConv2D(kernel_size=(3, 3), padding="same"),
        layers.ReLU(),
    ),
    "model014_depthwise_1x3x3x8_valid": sequential(
        (13, 15, 8),
        layers.DepthwiseConv2D(kernel_size=(3, 3)),
        layers.ReLU(),
    ),
    "model015_depthwise_global_12x12x45": depthwise((12, 12, 45), (12, 12)),
}
