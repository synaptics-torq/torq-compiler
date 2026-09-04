# Copyright 2025-2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Re-exports the ONNX quantization helpers from ``torq.lab.quantize_onnx``."""

from torq.lab.quantize_onnx import (
    RandomCalibrationDataReader,
    _ONNX_QUANTIZATION_OPTIONS,
    _parse_quant_dtype,
    _parse_quant_format,
    add_onnx_quantization_args,
    add_onnx_quantization_options,
    convert_qdq_to_full_integer,
    get_input_specs,
    is_model_quantized,
    quantize_onnx_model,
    quantize_onnx_static,
    quantize_onnx_static_from_model,
)

__all__ = [
    "RandomCalibrationDataReader",
    "add_onnx_quantization_args",
    "add_onnx_quantization_options",
    "convert_qdq_to_full_integer",
    "get_input_specs",
    "is_model_quantized",
    "quantize_onnx_model",
    "quantize_onnx_static",
    "quantize_onnx_static_from_model",
]
