# Copyright 2026 SYNAPTICS Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Convert a TFLite model to TOSA MLIR text that the Torq compiler parses.

tosa-converter-for-tflite is built against its own LLVM. Its ``--text`` output
uses that LLVM's TOSA syntax, which the compiler's MLIR no longer parses since
IREE 3.13 (TOSA ops print their properties as ``pad([...])``, not in an attribute
dictionary). Its bytecode stays readable across MLIR versions, so convert to
bytecode and print it with the compiler's own MLIR.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


def write_mlir_text_from_bytecode(bytecode_path: PathLike, text_path: PathLike) -> None:
    """Print an MLIR bytecode file as text with the compiler's MLIR."""
    from iree.compiler import ir

    with ir.Context():
        module = ir.Module.parse(Path(bytecode_path).read_bytes())
        Path(text_path).write_text(str(module))


def run_tflite_to_tosa(
    tflite_model_path: PathLike,
    output_mlir_path: PathLike,
    tool: str = "tosa-converter-for-tflite",
    **run_kwargs,
) -> subprocess.CompletedProcess:
    """Run the converter on a TFLite model and write TOSA MLIR text.

    ``run_kwargs`` are passed to ``subprocess.run`` (e.g. ``capture_output``,
    ``timeout``, ``check``). The text is written only when the converter
    succeeds; the converter's result is returned either way.
    """
    with tempfile.TemporaryDirectory(prefix="torq_tflite_to_tosa_") as tmp:
        bytecode = Path(tmp) / "model.tosa.mlirbc"
        result = subprocess.run(
            [tool, str(tflite_model_path), "--bytecode", "-o", str(bytecode)], **run_kwargs
        )
        if result.returncode == 0:
            write_mlir_text_from_bytecode(bytecode, output_mlir_path)
    return result
