# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""Weight-only quantization for ONNX models.

Supports int8 (asymmetric, ORT-matching) and int4 (signed) block quantization
of MatMul weight initializers.  Optionally dequantizes back to a single bf16
model suitable for IREE compilation.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, numpy_helper

from torq.lab.logging import add_logging_args, configure_logging
from ._config import LayerQuantConfig, QuantizationConfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class QuantizedWeight:
    """Result of quantizing a single weight tensor."""

    quantized: np.ndarray  # uint8 (int8) or int8 (int4), shape (K, N)
    scales: np.ndarray  # fp32, shape (n_blocks, N)
    zero_points: np.ndarray  # uint8 (int8) or int8 (int4), shape (n_blocks, N)
    block_size: int
    bits: int  # 4 or 8
    original_shape: tuple[int, ...]  # (K, N)


# ---------------------------------------------------------------------------
# Core quantization functions
# ---------------------------------------------------------------------------


def quantize_int8_asymmetric(
    weight: np.ndarray,
    block_size: int = 32,
) -> QuantizedWeight:
    """ORT-matching asymmetric uint8 block quantization.

    Blocks are along the K dimension (axis 0) of a (K, N) weight matrix.
    Internally the weight is transposed to (N, K) and reshaped into
    (N, n_blocks, block_size) — the same layout ORT ``MatMulNBits`` uses.

    Parameters
    ----------
    weight : (K, N) fp32 array
    block_size : number of elements per quantization block

    Returns
    -------
    QuantizedWeight with int8 quantized values, fp32 scales, int8 zero points.
    """
    weight = weight.astype(np.float32)
    K, N = weight.shape
    n_blocks = (K + block_size - 1) // block_size

    # Transpose to (N, K) and pad K to multiple of block_size
    w_t = weight.T
    pad_k = n_blocks * block_size - K
    if pad_k > 0:
        w_t = np.pad(w_t, ((0, 0), (0, pad_k)))
    w_blocked = w_t.reshape(N, n_blocks, block_size)

    w_min = w_blocked.min(axis=2, keepdims=True)
    w_max = w_blocked.max(axis=2, keepdims=True)

    scale = (w_max - w_min) / 255.0
    scale = np.maximum(scale, 1e-10)
    # Compute in uint8 space then convert to int8 (subtract 128)
    zp_u8 = np.round(-w_min / scale).clip(0, 255).astype(np.uint8)
    q_u8 = np.round(w_blocked / scale + zp_u8.astype(np.float32)).clip(0, 255).astype(
        np.uint8
    )
    # Convert to int8: int8 = uint8 - 128
    # (int8_q - int8_zp) * scale = (uint8_q - uint8_zp) * scale
    q = (q_u8.astype(np.int16) - 128).astype(np.int8)
    zp = (zp_u8.astype(np.int16) - 128).astype(np.int8)

    # Reshape back to (K, N) storage layout for DequantizeLinear axis=0
    q_kn = q.reshape(N, n_blocks * block_size)[:, :K].T.copy()  # (K, N) int8
    s_kn = scale.reshape(N, n_blocks).T.copy()  # (n_blocks, N) fp32
    zp_kn = zp.reshape(N, n_blocks).T.copy()  # (n_blocks, N) int8

    return QuantizedWeight(
        quantized=q_kn,
        scales=s_kn,
        zero_points=zp_kn,
        block_size=block_size,
        bits=8,
        original_shape=(K, N),
    )


def quantize_int4_signed(
    weight: np.ndarray,
    block_size: int = 32,
) -> QuantizedWeight:
    """Signed int4 [-8, 7] block quantization.

    Same block layout as :func:`quantize_int8_asymmetric` but with 15 levels
    instead of 255.

    Parameters
    ----------
    weight : (K, N) fp32 array
    block_size : number of elements per quantization block

    Returns
    -------
    QuantizedWeight with int8 quantized values (range [-8, 7]), fp32 scales,
    int8 zero points.
    """
    weight = weight.astype(np.float32)
    K, N = weight.shape
    n_blocks = (K + block_size - 1) // block_size

    w_t = weight.T
    pad_k = n_blocks * block_size - K
    if pad_k > 0:
        w_t = np.pad(w_t, ((0, 0), (0, pad_k)))
    w_blocked = w_t.reshape(N, n_blocks, block_size)

    w_min = w_blocked.min(axis=2, keepdims=True)
    w_max = w_blocked.max(axis=2, keepdims=True)

    scale = (w_max - w_min) / 15.0
    scale = np.maximum(scale, 1e-10)
    zp = np.round(-8.0 - w_min / scale).clip(-8, 7).astype(np.int8)
    q = np.round(w_blocked / scale + zp.astype(np.float32)).clip(-8, 7).astype(
        np.int8
    )

    q_kn = q.reshape(N, n_blocks * block_size)[:, :K].T.copy()
    s_kn = scale.reshape(N, n_blocks).T.copy()
    zp_kn = zp.reshape(N, n_blocks).T.copy()

    return QuantizedWeight(
        quantized=q_kn,
        scales=s_kn,
        zero_points=zp_kn,
        block_size=block_size,
        bits=4,
        original_shape=(K, N),
    )


def quantize_weight(
    weight: np.ndarray,
    bits: int,
    block_size: int = 32,
) -> QuantizedWeight:
    """Quantize *weight* to the given precision.

    Parameters
    ----------
    weight : (K, N) fp32 array
    bits : 4 or 8
    block_size : block size for block quantization
    """
    if bits == 8:
        return quantize_int8_asymmetric(weight, block_size)
    if bits == 4:
        return quantize_int4_signed(weight, block_size)
    raise ValueError(f"Unsupported bits={bits}; expected 4 or 8")


def dequantize_weight(qw: QuantizedWeight) -> np.ndarray:
    """Dequantize a :class:`QuantizedWeight` back to fp32.

    Uses bf16-truncated scales so the result matches what the hardware
    computes at runtime.

    Returns (K, N) fp32 array.
    """
    K, N = qw.original_shape
    bs = qw.block_size
    n_blocks = qw.scales.shape[0]

    # Truncate scales to bf16 precision
    scales_bf16 = _fp32_to_bf16_precision(qw.scales)

    q = qw.quantized.astype(np.float32)  # (K, N)
    zp = qw.zero_points.astype(np.float32)  # (n_blocks, N)
    deq = np.zeros((K, N), dtype=np.float32)
    for b in range(n_blocks):
        start = b * bs
        end = min(start + bs, K)
        deq[start:end, :] = (q[start:end, :] - zp[b, :]) * scales_bf16[b, :]
    return deq


# ---------------------------------------------------------------------------
# bf16 helpers
# ---------------------------------------------------------------------------


def _fp32_to_bf16_precision(arr: np.ndarray) -> np.ndarray:
    """Round fp32 values to bf16 precision (truncation)."""
    u32 = arr.view(np.uint32)
    truncated = (u32 & np.uint32(0xFFFF0000)).view(np.float32)
    return truncated.copy()


def _fp32_to_bf16_raw(arr: np.ndarray) -> bytes:
    """Convert fp32 array to raw bf16 bytes (truncation)."""
    u32 = arr.astype(np.float32).view(np.uint32)
    u16 = (u32 >> 16).astype(np.uint16)
    return u16.tobytes()


def _pack_int4(values: np.ndarray) -> bytes:
    """Pack int4 values (range [-8, 7]) into bytes, 2 values per byte.

    ONNX INT4 packing: low nibble first, high nibble second.
    Values are stored as unsigned nibbles: ``(val & 0xF)``.
    """
    flat = values.flatten().astype(np.int8)
    # Pad to even length
    if len(flat) % 2 != 0:
        flat = np.append(flat, np.int8(0))
    low = flat[0::2].astype(np.uint8) & 0x0F
    high = flat[1::2].astype(np.uint8) & 0x0F
    packed = low | (high << 4)
    return packed.tobytes()


def _make_int4_initializer(
    name: str, values: np.ndarray, shape: tuple[int, ...]
) -> TensorProto:
    """Create an ONNX TensorProto with INT4 dtype from int8 values [-8, 7]."""
    init = TensorProto()
    init.name = name
    init.dims[:] = list(shape)
    init.data_type = TensorProto.INT4
    init.raw_data = _pack_int4(values)
    return init


# ---------------------------------------------------------------------------
# WeightQuantizer — operates on an ONNX model
# ---------------------------------------------------------------------------


class WeightQuantizer:
    """Quantize MatMul weights in an ONNX model.

    Parameters
    ----------
    model_path : path to the input fp32 ONNX model
    output_path : path to write the quantized model
    skip_layers : layer name substrings to skip (e.g. ``["lm_head"]``)
    """

    def __init__(
        self,
        model_path: str | os.PathLike | None = None,
        output_path: str | os.PathLike | None = None,
        skip_layers: list[str] | None = None,
    ):
        self.model_path = Path(model_path) if model_path is not None else None
        self.output_path = Path(output_path) if output_path is not None else None
        self.skip_layers = skip_layers or []
        self._model: onnx.ModelProto | None = None

    # --- public API ----------------------------------------------------------

    def quantize_uniform(
        self,
        bits: int,
        block_size: int = 32,
        dequantize_weights: bool = False,
    ) -> Path:
        """Quantize all MatMul weights uniformly.

        Parameters
        ----------
        bits : 4, 8, or 16 (16 → bf16 conversion only, no quantization)
        block_size : block size for int4/int8 quantization
        dequantize_weights : if True, dequantize back and output a bf16 model
        """
        config = QuantizationConfig.uniform(bits, block_size)
        return self.quantize(config, dequantize_weights=dequantize_weights)

    def quantize(
        self,
        config: QuantizationConfig,
        dequantize_weights: bool = False,
    ) -> Path:
        """Main quantization entry point (file I/O).

        Parameters
        ----------
        config : quantization configuration
        dequantize_weights : if True, dequantize all weights and produce a
            single bf16 model ready for compilation
        """
        model = self._load_model()
        self.quantize_model(model, config, dequantize_weights=dequantize_weights)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save(model, str(self.output_path))
        size_mb = self.output_path.stat().st_size / 1e6
        logger.info("Saved quantized model to %s (%.1f MB)", self.output_path, size_mb)
        return self.output_path

    def quantize_model(
        self,
        model: onnx.ModelProto,
        config: QuantizationConfig,
        dequantize_weights: bool = False,
    ) -> onnx.ModelProto:
        """Quantize MatMul weights in an in-memory model (no file I/O).

        Parameters
        ----------
        model : the fp32 ONNX model to quantize
        config : quantization configuration
        dequantize_weights : if True, dequantize all weights and produce a
            single bf16 model ready for compilation
        """
        matmul_info = self._find_matmul_weights(model)
        logger.info("Found %d MatMul weight layers", len(matmul_info))

        # Check if this is a bf16-only conversion (all layers bits=16)
        all_bf16 = all(
            config.get(node.name).bits == 16 for node, _, _ in matmul_info
        )

        if dequantize_weights or all_bf16:
            self._quantize_dequantize_bf16(model, matmul_info, config)
        else:
            self._quantize_to_dql(model, matmul_info, config)
        return model

    # --- internals -----------------------------------------------------------

    def _load_model(self) -> onnx.ModelProto:
        if self._model is None:
            logger.info("Loading model from %s", self.model_path)
            self._model = onnx.load(str(self.model_path), load_external_data=True)
        return self._model

    def _should_skip(self, node_name: str) -> bool:
        return any(s in node_name for s in self.skip_layers)

    def _find_matmul_weights(
        self, model: onnx.ModelProto
    ) -> list[tuple[onnx.NodeProto, str, onnx.TensorProto]]:
        """Find MatMul nodes whose second input is a 2-D fp32 initializer.

        Returns list of ``(node, weight_name, initializer)``.
        """
        init_map = {i.name: i for i in model.graph.initializer}
        results = []
        for node in model.graph.node:
            if node.op_type != "MatMul":
                continue
            if self._should_skip(node.name):
                logger.debug("Skipping %s (matches skip_layers)", node.name)
                continue
            # Weight is typically the second input
            for inp in node.input:
                if inp in init_map:
                    init = init_map[inp]
                    if len(init.dims) == 2 and init.data_type == TensorProto.FLOAT:
                        results.append((node, inp, init))
                        break
        return results

    def _quantize_to_dql(
        self,
        model: onnx.ModelProto,
        matmul_info: list[tuple[onnx.NodeProto, str, onnx.TensorProto]],
        config: QuantizationConfig,
    ) -> None:
        """Replace MatMul weights with DequantizeLinear + MatMul.

        Produces int8/int4 weights with bf16 scales.
        bf16 (bits=16) layers are left as-is.
        """
        # Ensure opset >= 21 for DequantizeLinear with block_size
        _ensure_opset(model, 21)
        init_map = {i.name: i for i in model.graph.initializer}
        nodes_to_add = []

        counts = {"int4": 0, "int8": 0, "bf16": 0}

        for node, weight_name, init in matmul_info:
            layer_cfg = config.get(node.name)
            if layer_cfg.bits == 16:
                counts["bf16"] += 1
                continue

            fp32_w = numpy_helper.to_array(init).astype(np.float32)
            qw = quantize_weight(fp32_w, layer_cfg.bits, layer_cfg.block_size)
            tag = "int4" if layer_cfg.bits == 4 else "int8"
            counts[tag] += 1

            # Create initializer names
            q_name = f"{weight_name}_quantized"
            s_name = f"{weight_name}_scales"
            zp_name = f"{weight_name}_zero_points"

            # Quantized weight initializer
            if layer_cfg.bits == 4:
                q_init = _make_int4_initializer(q_name, qw.quantized, qw.quantized.shape)
            else:
                q_init = numpy_helper.from_array(qw.quantized, name=q_name)

            # Scales as bf16
            s_init = TensorProto()
            s_init.name = s_name
            s_init.dims[:] = list(qw.scales.shape)
            s_init.data_type = TensorProto.BFLOAT16
            s_init.raw_data = _fp32_to_bf16_raw(qw.scales)

            # Zero points
            if layer_cfg.bits == 4:
                zp_init = _make_int4_initializer(zp_name, qw.zero_points, qw.zero_points.shape)
            else:
                zp_init = numpy_helper.from_array(qw.zero_points, name=zp_name)

            # Remove old weight initializer
            if weight_name in init_map:
                for i, existing in enumerate(model.graph.initializer):
                    if existing.name == weight_name:
                        model.graph.initializer.remove(existing)
                        break
                del init_map[weight_name]

            # Add new initializers
            model.graph.initializer.extend([q_init, s_init, zp_init])

            # Create DequantizeLinear node
            dql_output = f"{weight_name}_dequantized"
            dql_node = onnx.helper.make_node(
                "DequantizeLinear",
                inputs=[q_name, s_name, zp_name],
                outputs=[dql_output],
                name=f"{node.name}/DequantizeLinear",
                axis=0,
                block_size=qw.block_size,
            )
            nodes_to_add.append(dql_node)

            # Rewire MatMul to use dequantized output
            for i, inp in enumerate(node.input):
                if inp == weight_name:
                    node.input[i] = dql_output

        # Insert DQL nodes before their consumers
        for dql in nodes_to_add:
            model.graph.node.insert(0, dql)

        logger.info(
            "Quantized layers: %d int4, %d int8, %d bf16 (skipped)",
            counts["int4"],
            counts["int8"],
            counts["bf16"],
        )

        # Convert remaining fp32 initializers and I/O to bf16, int64 to int32
        self._convert_non_weight_to_bf16(model)

    def _quantize_dequantize_bf16(
        self,
        model: onnx.ModelProto,
        matmul_info: list[tuple[onnx.NodeProto, str, onnx.TensorProto]],
        config: QuantizationConfig,
    ) -> None:
        """Quantize → dequantize → replace weights inline, then convert to bf16.

        The result is a single bf16 model with no DQL nodes — weights have the
        quantization error baked in but are stored as bf16 constants.
        """
        counts = {"int4": 0, "int8": 0, "bf16": 0}

        for node, weight_name, init in matmul_info:
            layer_cfg = config.get(node.name)
            if layer_cfg.bits == 16:
                counts["bf16"] += 1
                continue

            fp32_w = numpy_helper.to_array(init).astype(np.float32)
            qw = quantize_weight(fp32_w, layer_cfg.bits, layer_cfg.block_size)
            deq_w = dequantize_weight(qw)
            tag = "int4" if layer_cfg.bits == 4 else "int8"
            counts[tag] += 1

            # Replace initializer data with dequantized fp32
            for existing in model.graph.initializer:
                if existing.name == weight_name:
                    new_init = numpy_helper.from_array(deq_w, name=weight_name)
                    existing.CopyFrom(new_init)
                    break

        logger.info(
            "Dequantized layers: %d int4, %d int8, %d bf16 (unchanged)",
            counts["int4"],
            counts["int8"],
            counts["bf16"],
        )

        # Convert entire model fp32 → bf16 and int64 → int32
        self._convert_to_bf16(model)

    @staticmethod
    def _fix_slice_int64max(model: onnx.ModelProto) -> int:
        """Replace INT64_MAX in Slice 'ends' with actual dimension sizes.

        ONNX allows ``ends=[INT64_MAX]`` to mean "to the end of the axis",
        but IREE's ONNX importer cannot handle unbounded slice bounds.
        This pass resolves them to concrete values via shape inference.

        Must be called **before** int64→int32 conversion (INT64_MAX truncated
        to int32 produces garbage values).

        Returns the number of Slice nodes fixed.
        """
        from onnx import shape_inference

        INT64_MAX = 9223372036854775807

        model_inferred = shape_inference.infer_shapes(model)

        # Build shape map from inferred model
        shapes: dict[str, list[int]] = {}
        for v in (
            list(model_inferred.graph.value_info)
            + list(model_inferred.graph.input)
            + list(model_inferred.graph.output)
        ):
            if v.type.tensor_type.shape.dim:
                shapes[v.name] = [
                    d.dim_value for d in v.type.tensor_type.shape.dim
                ]

        init_map = {i.name: idx for idx, i in enumerate(model.graph.initializer)}

        fixed = 0
        for node in model.graph.node:
            if node.op_type != "Slice":
                continue
            inputs = list(node.input)
            if len(inputs) < 4:
                continue

            ends_name = inputs[2]
            axes_name = inputs[3]
            data_name = inputs[0]

            if ends_name not in init_map or axes_name not in init_map:
                continue

            ends_arr = numpy_helper.to_array(
                model.graph.initializer[init_map[ends_name]]
            )
            axes_arr = numpy_helper.to_array(
                model.graph.initializer[init_map[axes_name]]
            )

            new_ends = ends_arr.copy()
            needs_fix = False
            for i, (end_val, axis_val) in enumerate(zip(ends_arr, axes_arr)):
                if end_val == INT64_MAX:
                    shape = shapes.get(data_name)
                    if shape and len(shape) > axis_val and shape[axis_val] > 0:
                        new_ends[i] = shape[axis_val]
                        needs_fix = True

            if needs_fix:
                new_tensor = numpy_helper.from_array(new_ends, name=ends_name)
                model.graph.initializer[init_map[ends_name]].CopyFrom(new_tensor)
                fixed += 1

        if fixed:
            logger.info(
                "Fixed %d Slice node(s): replaced INT64_MAX ends with actual dims",
                fixed,
            )
        return fixed

    def _convert_to_bf16(self, model: onnx.ModelProto) -> None:
        """Convert fp32 model to bf16 inline using the proper dtype converter.

        Delegates to :mod:`torq.lab.model_tools.dtype_conversion.onnx` which
        uses onnx-graphsurgeon with shape inference and proper handling of
        Slice/Reshape/etc. inputs (kept as int64 per ONNX spec).
        """
        from torq.lab.model_tools.dtype_conversion.onnx import (
            FP32Converter,
            Int64Converter,
        )

        # Fix INT64_MAX in Slice ends before int64→int32 conversion
        self._fix_slice_int64max(model)

        fp32_conv = FP32Converter("bf16", convert_io=True)
        converted = fp32_conv.convert_model(model)

        int64_conv = Int64Converter("int32", convert_io=True)
        converted = int64_conv.convert_model(converted)

        model.CopyFrom(converted)
        logger.info("Converted model to bf16 + int32")

    def _convert_non_weight_to_bf16(self, model: onnx.ModelProto) -> None:
        """Convert non-quantized parts of the model to bf16/int32.

        Delegates to :mod:`torq.lab.model_tools.dtype_conversion.onnx` which
        uses onnx-graphsurgeon with shape inference and proper handling of
        Slice/Reshape/etc. inputs (kept as int64 per ONNX spec).
        Initializers that are already int8/uint8/int4/bf16 (quantized weights,
        scales, zero points) are preserved by the converter since they don't
        match the fp32 source dtype.
        """
        from torq.lab.model_tools.dtype_conversion.onnx import (
            FP32Converter,
            Int64Converter,
        )

        # Fix INT64_MAX in Slice ends before int64→int32 conversion
        self._fix_slice_int64max(model)

        fp32_conv = FP32Converter("bf16", convert_io=True)
        converted = fp32_conv.convert_model(model)

        int64_conv = Int64Converter("int32", convert_io=True)
        converted = int64_conv.convert_model(converted)

        model.CopyFrom(converted)
        logger.info("Converted non-weight tensors and I/O to bf16 + int32")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_opset(model: onnx.ModelProto, min_opset: int) -> None:
    """Ensure the default ONNX opset is at least *min_opset*."""
    for opset in model.opset_import:
        if opset.domain == "" or opset.domain == "ai.onnx":
            if opset.version < min_opset:
                opset.version = min_opset
            return
    model.opset_import.append(onnx.helper.make_opsetid("", min_opset))


# ---------------------------------------------------------------------------
# Lab entry points (ModelProto / file I/O)
# ---------------------------------------------------------------------------


def onnx_weights_quantize(
    model: onnx.ModelProto,
    *,
    bits: int = 8,
    block_size: int = 32,
    config: QuantizationConfig | None = None,
    dequantize_weights: bool = False,
    skip_layers: list[str] | None = None,
) -> onnx.ModelProto:
    """Weight-only quantize an in-memory ONNX model.

    Quantizes MatMul weight initializers to int8 (asymmetric) or int4
    (signed) with per-block bf16 scales, emitted as ``DequantizeLinear``
    nodes; the remaining fp32 tensors and I/O are converted to bf16 and
    int64 to int32.  With ``dequantize_weights`` (or an all-bf16 config)
    the quantization error is baked into bf16 weights instead and no DQL
    nodes are added.

    Args:
        model: The fp32 ONNX model to quantize.
        bits: Uniform bit-width (4, 8, or 16 → bf16 only) when *config* is
            omitted.
        block_size: Block size for int4/int8 block quantization.
        config: Per-layer quantization configuration; overrides *bits* and
            *block_size*.
        dequantize_weights: Dequantize weights back and emit a single bf16
            model ready for compilation (no DQL nodes).
        skip_layers: Layer name substrings to skip.

    Returns:
        The quantized ONNX model.
    """
    if config is None:
        config = QuantizationConfig.uniform(bits, block_size)
    quantizer = WeightQuantizer(skip_layers=skip_layers)
    quantizer.quantize_model(model, config, dequantize_weights=dequantize_weights)
    return model


def onnx_weights_quantize_file(
    model_input: str | os.PathLike,
    model_output: str | os.PathLike,
    *,
    bits: int = 8,
    block_size: int = 32,
    config: QuantizationConfig | None = None,
    dequantize_weights: bool = False,
    skip_layers: list[str] | None = None,
) -> Path:
    """Weight-only quantize an ONNX model file.

    Thin file-I/O wrapper around :func:`onnx_weights_quantize`.

    Args:
        model_input: Path to the fp32 ONNX model.
        model_output: Path where the quantized ONNX model will be written.
        bits: Uniform bit-width (4, 8, or 16 → bf16 only) when *config* is
            omitted.
        block_size: Block size for int4/int8 block quantization.
        config: Per-layer quantization configuration; overrides *bits* and
            *block_size*.
        dequantize_weights: Dequantize weights back and emit a single bf16
            model ready for compilation (no DQL nodes).
        skip_layers: Layer name substrings to skip.

    Returns:
        The path to the written quantized model.
    """
    model = onnx.load(str(model_input), load_external_data=True)
    onnx_weights_quantize(
        model,
        bits=bits,
        block_size=block_size,
        config=config,
        dequantize_weights=dequantize_weights,
        skip_layers=skip_layers,
    )
    model_output = Path(model_output)
    model_output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(model_output))
    size_mb = model_output.stat().st_size / 1e6
    logger.info("Saved quantized model to %s (%.1f MB)", model_output, size_mb)
    return model_output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def add_weights_quantize_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input fp32 ONNX model path",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output quantized ONNX model path",
    )
    parser.add_argument(
        "--bits",
        type=int,
        choices=[4, 8, 16],
        default=None,
        help="Uniform quantization bit-width (4=int4, 8=int8, 16=bf16). "
        "Ignored when --config is provided.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=32,
        help="Block size for block quantization (default: %(default)s)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to quantization config JSON (from sensitivity analysis). "
        "Overrides --bits for per-layer mixed quantization.",
    )
    parser.add_argument(
        "--dequantize-weights",
        action="store_true",
        help="Dequantize weights and output a single bf16 model "
        "ready for IREE compilation (no DQL nodes).",
    )
    parser.add_argument(
        "--skip-layers",
        type=str,
        nargs="*",
        default=[],
        help="Layer name substrings to skip (e.g. lm_head)",
    )
    add_logging_args(parser)


def weights_quantize_from_args(args: argparse.Namespace) -> Path:
    configure_logging(args.logging)

    model_path = Path(args.input)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if args.config:
        config = QuantizationConfig.load(args.config)
        logger.info("Loaded per-layer config from %s", args.config)
    elif args.bits is not None:
        config = QuantizationConfig.uniform(args.bits, args.block_size)
        logger.info(
            "Uniform %d-bit quantization, block_size=%d", args.bits, args.block_size
        )
    else:
        raise ValueError("Either --bits or --config is required for quantize")

    return onnx_weights_quantize_file(
        args.input,
        args.output,
        config=config,
        dequantize_weights=args.dequantize_weights,
        skip_layers=args.skip_layers,
    )
