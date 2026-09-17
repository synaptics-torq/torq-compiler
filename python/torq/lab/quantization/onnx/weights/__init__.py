# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""ONNX weight-only (int4/int8/bf16) quantization of MatMul weights."""

from ._analysis import (
    LayerSensitivityAnalyzer,
    add_weights_analyze_args,
    weights_analyze_from_args,
)
from ._config import (
    LayerQuantConfig,
    QuantizationConfig,
    SensitivityResult,
    SensitivityResults,
)
from ._quantization import (
    WeightQuantizer,
    add_weights_quantize_args,
    dequantize_weight,
    onnx_weights_quantize,
    onnx_weights_quantize_file,
    quantize_int4_signed,
    quantize_int8_asymmetric,
    quantize_weight,
    weights_quantize_from_args,
)

__all__ = [
    "LayerQuantConfig",
    "QuantizationConfig",
    "SensitivityResult",
    "SensitivityResults",
    "WeightQuantizer",
    "quantize_int8_asymmetric",
    "quantize_int4_signed",
    "quantize_weight",
    "dequantize_weight",
    "onnx_weights_quantize",
    "onnx_weights_quantize_file",
    "add_weights_quantize_args",
    "weights_quantize_from_args",
    "LayerSensitivityAnalyzer",
    "add_weights_analyze_args",
    "weights_analyze_from_args",
]
