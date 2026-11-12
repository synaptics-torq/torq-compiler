"""Keras test models: Softmax."""

from torq.testing.keras_builder import layers, sequential

CASES = {
    "model089_softmax_inp1x1916x2": sequential((1916, 2), layers.Softmax()),
    # int8 softmax over the last axis, as in YOLO: the 16 bins of the yolov8 DFL box decoding, and
    # the 100 keys of the yolo26 attention.
    "softmax_inp4x64x16": sequential((4, 64, 16), layers.Softmax()),
    "softmax_inp2x8x100": sequential((2, 8, 100), layers.Softmax()),
}

CONFIG = {
    # Softmax runs on the slices only in bf16.
    "model089_softmax_inp1x1916x2": {"compiler_options": ["--torq-convert-dtypes"]},
    # The int16 softmax is another TFLite kernel, not covered by these cases.
    "softmax_inp4x64x16": {"quantize": "int8"},
    "softmax_inp2x8x100": {"quantize": "int8"},
}
