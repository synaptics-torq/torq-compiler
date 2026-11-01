"""Keras test models: DepthwiseConv2D followed by ReLU, and a global depthwise conv.

Each case name says what it tests: kernel, strides, padding, input shape (H x W x C) and the
activation after the conv, e.g. dw_k3x3_s1x1_valid_6x42x16_relu.
"""

from torq.testing.keras_builder import layers, sequential, depthwise

# int8 only: the TFLite converter fails to quantize DepthwiseConv2D to int16 ("Quantized dimension
# for tensor property and quantization parameters do not match").
QUANTIZE = ("int8",)

# (input H, W, C), kernel (h, w), padding; each followed by ReLU
_LAYERS = [
    ((6, 42, 16), (3, 3), "valid"),
    ((6, 4, 16), (3, 3), "valid"),
    ((6, 6, 2), (3, 3), "valid"),
    ((6, 6, 2), (3, 3), "same"),
    ((13, 15, 8), (3, 3), "valid"),
]


def _name(shape, kernel, padding, after=""):
    h, w, c = shape
    return f"dw_k{kernel[0]}x{kernel[1]}_s1x1_{padding}_{h}x{w}x{c}{after}"


CASES = {
    # a kernel as large as the frame
    _name((12, 12, 45), (12, 12), "valid"): depthwise((12, 12, 45), (12, 12)),
}
for _shape, _kernel, _padding in _LAYERS:
    _key = _name(_shape, _kernel, _padding, "_relu")
    if _key in CASES:
        raise ValueError(f"depthwise: {_key} is listed twice")
    CASES[_key] = sequential(_shape, layers.DepthwiseConv2D(kernel_size=_kernel, padding=_padding),
                             layers.ReLU())
