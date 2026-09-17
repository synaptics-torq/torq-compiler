# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ONNX dynamic (int8, activation-float) quantization via onnxruntime."""

from ._analysis import (
    add_dynamic_analyze_args,
    analyze_dynamic_quantization,
    dynamic_analyze_from_args,
    summarize_dynamic_quantization,
)
from ._quantization import (
    add_dynamic_quantize_args,
    dynamic_quantize_from_args,
    onnx_dynamic_quantize,
    onnx_dynamic_quantize_file,
)

__all__ = [
    "add_dynamic_quantize_args",
    "dynamic_quantize_from_args",
    "onnx_dynamic_quantize",
    "onnx_dynamic_quantize_file",
    "add_dynamic_analyze_args",
    "dynamic_analyze_from_args",
    "analyze_dynamic_quantization",
    "summarize_dynamic_quantization",
]
