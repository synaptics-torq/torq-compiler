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


def _name(shape, kernel, padding, after="", strides=(1, 1)):
    h, w, c = shape
    return f"dw_k{kernel[0]}x{kernel[1]}_s{strides[0]}x{strides[1]}_{padding}_{h}x{w}x{c}{after}"


# A kernel as large as the (padded) frame with stride 2: one output, which the stride does not
# change. Without CSS, these did not compile (synaptics-torq/torq-compiler-dev#2806).
_GLOBAL_STRIDE2 = [
    ((3, 3, 8), (3, 3), "valid"),
    ((2, 2, 8), (3, 3), "same"),
    ((12, 12, 45), (12, 12), "valid"),
]


CASES = {
    # a kernel as large as the frame
    _name((12, 12, 45), (12, 12), "valid"): depthwise((12, 12, 45), (12, 12)),
}
for _shape, _kernel, _padding in _GLOBAL_STRIDE2:
    CASES[_name(_shape, _kernel, _padding, strides=(2, 2))] = depthwise(
        _shape, _kernel, (2, 2), _padding
    )
for _shape, _kernel, _padding in _LAYERS:
    _key = _name(_shape, _kernel, _padding, "_relu")
    if _key in CASES:
        raise ValueError(f"depthwise: {_key} is listed twice")
    CASES[_key] = sequential(_shape, layers.DepthwiseConv2D(kernel_size=_kernel, padding=_padding),
                             layers.ReLU())

# The stride-2 global cases must run on the slices.
_NSS_ONLY = {"compiler_options": ["--torq-disable-host", "--torq-disable-css"]}
CONFIG = {_name(shape, kernel, padding, strides=(2, 2)): _NSS_ONLY
          for shape, kernel, padding in _GLOBAL_STRIDE2}
