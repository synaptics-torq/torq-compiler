"""Keras test models: Conv2D and DepthwiseConv2D with stride (2, 1).

The compiler splits the kernel rows of a stride (2, 1) conv into an even and an odd phase. It then
runs a stride-1 conv on the even and odd input rows (synaptics-torq/torq-compiler-dev#2912).

Each case name gives the kernel, padding and input shape (H x W x C), e.g. s21_conv_k5x3_valid_11x11x4.
"""

from torq.testing.keras_builder import layers, sequential

QUANTIZE = ("int8",)
DEFAULT_CONFIG = {"compiler_options": ["--torq-disable-host", "--torq-disable-css"]}
_BIAS = {"bias_initializer": "random_uniform"}

CASES = {}
for _kh, _kw in [(3, 3), (3, 5), (5, 3), (5, 5), (2, 2), (4, 4)]:
    for _pad in ["same", "valid"]:
        for _h, _w in [(11, 11), (12, 12)]:
            CASES[f"s21_conv_k{_kh}x{_kw}_{_pad}_{_h}x{_w}x4"] = sequential(
                (_h, _w, 4), layers.Conv2D(8, (_kh, _kw), strides=(2, 1), padding=_pad, **_BIAS))
            CASES[f"s21_dw_k{_kh}x{_kw}_{_pad}_{_h}x{_w}x8"] = sequential(
                (_h, _w, 8), layers.DepthwiseConv2D((_kh, _kw), strides=(2, 1), padding=_pad,
                                                    **_BIAS))
