# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ONNX static (int8, calibration) quantization via onnxruntime."""

from ._analysis import (
    add_static_analyze_args,
    analyze_static_quantization,
    static_analyze_from_args,
)
from ._quantization import (
    CalibrationDataReader,
    ONNX_STATIC_QUANTIZATION_OPTIONS,
    QuantFormat,
    QuantType,
    RandomCalibrationDataReader,
    add_onnx_static_quantization_args,
    add_onnx_static_quantization_options,
    add_static_quant_flags,
    add_static_quantize_args,
    convert_qdq_to_full_integer,
    get_input_specs,
    is_model_quantized,
    onnx_static_quantize,
    onnx_static_quantize_file,
    parse_quant_dtype,
    parse_quant_format,
    quantize_static,
    static_quantize_from_args,
)

__all__ = [
    "add_static_quantize_args",
    "static_quantize_from_args",
    "add_static_analyze_args",
    "static_analyze_from_args",
    "analyze_static_quantization",
    "CalibrationDataReader",
    "ONNX_STATIC_QUANTIZATION_OPTIONS",
    "QuantFormat",
    "QuantType",
    "RandomCalibrationDataReader",
    "add_onnx_static_quantization_args",
    "add_onnx_static_quantization_options",
    "add_static_quant_flags",
    "add_static_quantize_args",
    "convert_qdq_to_full_integer",
    "get_input_specs",
    "is_model_quantized",
    "onnx_static_quantize",
    "onnx_static_quantize_file",
    "parse_quant_dtype",
    "parse_quant_format",
    "quantize_static",
    "static_quantize_from_args",
]
