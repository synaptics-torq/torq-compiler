# Copyright 2026 SYNAPTICS Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""
Workaround for quantized TFLite Abs until Arm's tosa-converter-for-tflite fix ships.

Upstream: https://gitlab.arm.com/tosa/tosa-converter-for-tflite/-/merge_requests/95

Stock tosa-converter-for-tflite maps tfl.abs -> tosa.abs directly. That fails when
input/output !quant.uniform parameters differ, because tosa.abs requires matching
element types. Even when scales are forced equal, tosa.abs on i8 is not legal in
the TOSA profile (i32 is required).

This module mirrors the upstream legalization:

  rescale to i32 (remove input zp) -> tosa.abs -> rescale to output quant

while still using the released converter binary:
  1. Snapshot Abs I/O scales/zero-points from the TFLite model.
  2. Temporarily make Abs output quant params match the input so conversion succeeds.
  3. Rewrite every resulting tosa.abs on i8 into the rescale -> abs(i32) -> rescale
     form using the snapshot from (1).

Remove this workaround once requirements pin a converter release that includes MR 95.
"""

from __future__ import annotations

import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np

logger = logging.getLogger("Torq-tflite-abs")

PathLike = Union[str, Path]


@dataclass(frozen=True)
class QuantizedAbsSite:
    """Per-tensor Abs quantization parameters from the original TFLite model."""

    shape: Tuple[int, ...]
    in_scale: float
    in_zp: int
    out_scale: float
    out_zp: int


def tosa_scale32_multiplier_shift(scale: float) -> Tuple[int, int]:
    """Encode a positive scale as TOSA scale32 (multiplier, shift).

    Matches the common encoding used in Torq testdata: scale ~= multiplier / 2^shift.
    For scale 1.0 this yields (1073741824, 30).
    """
    if scale == 0.0:
        return 0, 0
    significand, exp = np.frexp(float(scale))
    q = int(round(significand * (1 << 31)))
    if q == (1 << 31):
        q //= 2
        exp += 1
    shift = 31 - exp
    if shift < -128 or shift > 127:
        raise ValueError(f"TOSA scale32 shift {shift} out of i8 range for scale={scale}")
    return q, int(shift)


def _opcode_code(opcode) -> int:
    code = opcode.builtinCode
    return code if code else opcode.deprecatedBuiltinCode


def _is_per_tensor_quant(tensor) -> bool:
    q = tensor.quantization
    return (
        q is not None
        and q.scale is not None
        and len(q.scale) == 1
        and q.zeroPoint is not None
        and len(q.zeroPoint) == 1
    )


def collect_quantized_abs_sites(tflite_model_path: PathLike) -> List[QuantizedAbsSite]:
    """Return Abs sites with per-tensor quantized i8/i16 input and output."""
    from tensorflow.lite.python import schema_py_generated as schema_fb
    from tensorflow.lite.tools import flatbuffer_utils

    model = flatbuffer_utils.read_model_with_mutable_tensors(str(tflite_model_path))
    abs_code = schema_fb.BuiltinOperator.ABS
    sites: List[QuantizedAbsSite] = []

    for subgraph in model.subgraphs:
        for op in subgraph.operators:
            if _opcode_code(model.operatorCodes[op.opcodeIndex]) != abs_code:
                continue
            if len(op.inputs) != 1 or len(op.outputs) != 1:
                continue
            in_t = subgraph.tensors[op.inputs[0]]
            out_t = subgraph.tensors[op.outputs[0]]
            if not (_is_per_tensor_quant(in_t) and _is_per_tensor_quant(out_t)):
                continue
            if in_t.shape is None:
                continue
            sites.append(
                QuantizedAbsSite(
                    shape=tuple(int(d) for d in in_t.shape),
                    in_scale=float(in_t.quantization.scale[0]),
                    in_zp=int(in_t.quantization.zeroPoint[0]),
                    out_scale=float(out_t.quantization.scale[0]),
                    out_zp=int(out_t.quantization.zeroPoint[0]),
                )
            )
    return sites


def _force_abs_output_quant_to_match_input(tflite_model_path: PathLike, dest: Path) -> int:
    """Rewrite Abs output quant params to match input so the stock converter accepts Abs."""
    from tensorflow.lite.python import schema_py_generated as schema_fb
    from tensorflow.lite.tools import flatbuffer_utils

    model = flatbuffer_utils.read_model_with_mutable_tensors(str(tflite_model_path))
    abs_code = schema_fb.BuiltinOperator.ABS
    changed = 0

    for subgraph in model.subgraphs:
        for op in subgraph.operators:
            if _opcode_code(model.operatorCodes[op.opcodeIndex]) != abs_code:
                continue
            if len(op.inputs) != 1 or len(op.outputs) != 1:
                continue
            in_t = subgraph.tensors[op.inputs[0]]
            out_t = subgraph.tensors[op.outputs[0]]
            if not (_is_per_tensor_quant(in_t) and _is_per_tensor_quant(out_t)):
                continue
            out_t.quantization.scale = list(in_t.quantization.scale)
            out_t.quantization.zeroPoint = list(in_t.quantization.zeroPoint)
            changed += 1

    flatbuffer_utils.write_model(model, str(dest))
    return changed


_ABS_I8_RE = re.compile(
    r"(?P<indent>[ \t]*)"
    r"(?P<lhs>%[\w.\d]+)\s*=\s*tosa\.abs\s+(?P<arg>%[\w.\d]+)\s*"
    r":\s*\(tensor<(?P<shape>[\dx]+)xi8>\)\s*->\s*tensor<(?P=shape)xi8>"
)


def _emit_abs_i32_legalization(indent: str, lhs: str, arg: str, shape: str, site: QuantizedAbsSite) -> str:
    mult0, shift0 = tosa_scale32_multiplier_shift(1.0)
    mult1, shift1 = tosa_scale32_multiplier_shift(site.in_scale / site.out_scale)
    # Numeric SSA ids (%0) cannot take suffixes in MLIR; use a named prefix.
    tag = "abs_" + re.sub(r"[^0-9A-Za-z_]", "_", lhs.lstrip("%"))
    p = f"{indent}%{tag}"
    return "\n".join(
        [
            f'{p}_mult0 = "tosa.const"() <{{values = dense<{mult0}> : tensor<1xi32>}}> : () -> tensor<1xi32>',
            f'{p}_shift0 = "tosa.const"() <{{values = dense<{shift0}> : tensor<1xi8>}}> : () -> tensor<1xi8>',
            f'{p}_in_zp = "tosa.const"() <{{values = dense<{site.in_zp}> : tensor<1xi8>}}> : () -> tensor<1xi8>',
            f'{p}_zero_i32 = "tosa.const"() <{{values = dense<0> : tensor<1xi32>}}> : () -> tensor<1xi32>',
            f'{p}_mult1 = "tosa.const"() <{{values = dense<{mult1}> : tensor<1xi32>}}> : () -> tensor<1xi32>',
            f'{p}_shift1 = "tosa.const"() <{{values = dense<{shift1}> : tensor<1xi8>}}> : () -> tensor<1xi8>',
            f'{p}_out_zp = "tosa.const"() <{{values = dense<{site.out_zp}> : tensor<1xi8>}}> : () -> tensor<1xi8>',
            (
                f"{p}_centered = tosa.rescale {arg}, %{tag}_mult0, %{tag}_shift0, "
                f"%{tag}_in_zp, %{tag}_zero_i32 "
                f"{{input_unsigned = false, output_unsigned = false, per_channel = false, "
                f"rounding_mode = DOUBLE_ROUND, scale32 = true}} : "
                f"(tensor<{shape}xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) "
                f"-> tensor<{shape}xi32>"
            ),
            f"{p}_absv = tosa.abs %{tag}_centered : (tensor<{shape}xi32>) -> tensor<{shape}xi32>",
            (
                f"{indent}{lhs} = tosa.rescale %{tag}_absv, %{tag}_mult1, %{tag}_shift1, "
                f"%{tag}_zero_i32, %{tag}_out_zp "
                f"{{input_unsigned = false, output_unsigned = false, per_channel = false, "
                f"rounding_mode = DOUBLE_ROUND, scale32 = true}} : "
                f"(tensor<{shape}xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) "
                f"-> tensor<{shape}xi8>"
            ),
        ]
    )


def replace_i8_abs_with_rescale_legalization(mlir_text: str, sites: Sequence[QuantizedAbsSite]) -> str:
    """Replace tosa.abs on i8 with rescale -> abs(i32) -> rescale using Abs site metadata."""
    if not sites:
        return mlir_text

    site_iter = iter(sites)
    replacements = 0

    def _sub(match: re.Match) -> str:
        nonlocal replacements
        try:
            site = next(site_iter)
        except StopIteration as exc:
            raise RuntimeError(
                "Found more tosa.abs i8 ops in converted MLIR than quantized Abs sites "
                "recorded from the TFLite model"
            ) from exc
        shape_from_mlir = match.group("shape")
        expected = "x".join(str(d) for d in site.shape)
        if shape_from_mlir != expected:
            logger.warning(
                "Abs legalization shape mismatch: MLIR %s vs TFLite %s; using MLIR shape",
                shape_from_mlir,
                expected,
            )
        replacements += 1
        return _emit_abs_i32_legalization(
            match.group("indent"),
            match.group("lhs"),
            match.group("arg"),
            shape_from_mlir,
            site,
        )

    new_text = _ABS_I8_RE.sub(_sub, mlir_text)
    leftover = list(site_iter)
    if leftover:
        raise RuntimeError(
            f"Converted MLIR had fewer tosa.abs i8 ops than quantized Abs sites "
            f"({len(sites) - len(leftover)} replacements, {len(leftover)} sites unused)"
        )
    logger.info("Legalized %d quantized TFLite Abs op(s) via rescale->abs(i32)->rescale", replacements)
    return new_text


def convert_tflite_to_tosa_mlir_with_abs_workaround(
    tflite_model_path: PathLike,
    output_mlir_path: PathLike,
) -> None:
    """Run tosa-converter-for-tflite, applying the Abs workaround when needed."""
    import subprocess

    tflite_model_path = Path(tflite_model_path)
    output_mlir_path = Path(output_mlir_path)
    sites = collect_quantized_abs_sites(tflite_model_path)

    if not sites:
        subprocess.check_call(
            ["tosa-converter-for-tflite", "--text", str(tflite_model_path), "-o", str(output_mlir_path)]
        )
        return

    logger.info(
        "Applying quantized Abs workaround for %d op(s) "
        "(pending upstream tosa-converter-for-tflite MR 95)",
        len(sites),
    )
    with tempfile.TemporaryDirectory(prefix="torq_abs_workaround_") as tmp:
        patched = Path(tmp) / "abs_quant_matched.tflite"
        _force_abs_output_quant_to_match_input(tflite_model_path, patched)
        subprocess.check_call(
            ["tosa-converter-for-tflite", "--text", str(patched), "-o", str(output_mlir_path)]
        )

    mlir_text = output_mlir_path.read_text()
    output_mlir_path.write_text(replace_i8_abs_with_rescale_legalization(mlir_text, sites))
