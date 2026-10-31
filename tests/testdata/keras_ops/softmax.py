"""Keras test models: Softmax."""

from torq.testing.keras_builder import layers, sequential

CASES = {
    "model089_softmax_inp1x1916x2": sequential((1916, 2), layers.Softmax()),
}

CONFIG = {
    # Softmax runs on the slices only in bf16.
    "model089_softmax_inp1x1916x2": {"compiler_options": ["--torq-convert-dtypes"]},
}
