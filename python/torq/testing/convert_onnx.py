# Copyright 2025-2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Re-exports the ONNX dtype-conversion helpers from ``torq.lab.convert_onnx``."""

from torq.lab.convert_onnx import (
    _fix_batch_dimension_to_one,
    convert_fp32_to_bf16,
    convert_int64_to_int32,
    is_model_bf16,
    is_model_int32,
)

__all__ = [
    "convert_fp32_to_bf16",
    "convert_int64_to_int32",
    "is_model_bf16",
    "is_model_int32",
]
