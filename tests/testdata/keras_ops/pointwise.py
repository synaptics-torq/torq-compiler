"""Keras test models: 1x1 Conv2D."""

from torq.testing.keras_builder import conv2d

CASES = {
    "model016_pointwise_8x8x8x16_stride2x2": conv2d((8, 8, 16), 8, (1, 1), (2, 2), "same"),
    "model017_pointwise_8x8x9x16_stride2x2": conv2d((8, 9, 16), 8, (1, 1), (2, 2), "same"),
    "model018_pointwise_8x9x8x16_stride2x2": conv2d((9, 8, 16), 8, (1, 1), (2, 2), "same"),
    "model019_pointwise_8x9x9x16_stride2x2": conv2d((9, 9, 16), 8, (1, 1), (2, 2), "same"),
    "model113_pointwise_1x1x1024": conv2d((8, 8, 1024), 8, (1, 1), (1, 1), "same"),
    "model240_pointwise_inp1x4x4x16_8x1x1_valid_stride1x1": conv2d((4, 4, 16), 8, (1, 1)),
}
