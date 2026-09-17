# Copyright 2025-2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Re-exports the ONNX quantization helpers from ``torq.lab.quantization.onnx.static``."""

from torq.lab.quantization.onnx.static import (
    ONNX_STATIC_QUANTIZATION_OPTIONS,
    RandomCalibrationDataReader,
    add_onnx_static_quantization_args,
    add_onnx_static_quantization_options,
    convert_qdq_to_full_integer,
    get_input_specs,
    is_model_quantized,
    onnx_static_quantize,
    onnx_static_quantize_file,
    parse_quant_dtype,
    parse_quant_format,
)

__all__ = [
    "RandomCalibrationDataReader",
    "add_onnx_static_quantization_args",
    "add_onnx_static_quantization_options",
    "convert_qdq_to_full_integer",
    "get_input_specs",
    "is_model_quantized",
    "onnx_static_quantize",
    "onnx_static_quantize_file",
]
