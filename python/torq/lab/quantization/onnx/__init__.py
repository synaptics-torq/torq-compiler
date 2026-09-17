# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ONNX quantization modes: static (int8), dynamic (int8), weight-only (int4/int8/bf16), and fake dtype quantization.

:func:`quantize_onnx_model` is the file-based entry point shared by the
``torq-lab quantize`` / ``torq-quantize-model`` CLI and ``torq-gen-config
quantize``; the per-mode implementations live in the ``static``,
``dynamic``, and ``weights`` subpackages as ``onnx_<mode>_quantize``
(in-memory ModelProto) and ``onnx_<mode>_quantize_file`` (file path).
"""

from pathlib import Path

__all__ = ["quantize_onnx_model"]


def quantize_onnx_model(
    model_input,
    model_output,
    *,
    method: str = "static",
    **method_kwargs,
) -> Path:
    """Quantize an ONNX model file with the selected method.

    Args:
        model_input: Path to the fp32 ONNX model.
        model_output: Path where the quantized ONNX model will be written.
        method: Quantization mode: ``"static"`` (int8 with calibration),
            ``"dynamic"`` (int8 via onnxruntime, no calibration), or
            ``"weights"`` (weight-only int4/int8/bf16 MatMul quantization).
        **method_kwargs: Forwarded to the mode's
            ``onnx_<method>_quantize_file`` function. static: ``num_calib``,
            ``calibration_data_reader``, ``dataset``, ``per_channel``,
            ``full_integer``, ``quant_format``, ``quant_dtype``,
            ``quantize_only_ops``, ``quantize_only_nodes``, ``exclude_nodes``;
            dynamic: see
            :func:`torq.lab.quantization.onnx.dynamic.onnx_dynamic_quantize_file`;
            weights: ``bits``, ``block_size``, ``config``,
            ``dequantize_weights``, ``skip_layers``.

    Returns:
        The path to the written quantized model.
    """
    model_input = Path(model_input)
    model_output = Path(model_output)

    if method == "static":
        from .static import onnx_static_quantize_file

        onnx_static_quantize_file(model_input, model_output, **method_kwargs)
    elif method == "dynamic":
        from .dynamic import onnx_dynamic_quantize_file

        onnx_dynamic_quantize_file(model_input, model_output, **method_kwargs)
    elif method == "weights":
        from .weights import onnx_weights_quantize_file

        onnx_weights_quantize_file(model_input, model_output, **method_kwargs)
    else:
        raise ValueError(f"unknown ONNX quantization method: {method!r}")
    return model_output
