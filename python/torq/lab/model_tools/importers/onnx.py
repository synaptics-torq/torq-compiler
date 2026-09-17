# Copyright 2025-2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""ONNX-to-MLIR model import.

Source-model importing for the pipeline: converts an ONNX model to MLIR via
``iree.compiler.tools.import_onnx``. Layer/subgraph extraction lives in
:mod:`torq.lab.model_tools.extraction.onnx`.
"""

import subprocess
import sys
from pathlib import Path

import onnx

from torq.lab.logging import is_verbose

def convert_onnx_to_mlir(model_path, output_path, timeout=300):
    """Convert an ONNX model file to MLIR via ``iree.compiler.tools.import_onnx``.

    Validates the ONNX model first, then runs the import in a subprocess with
    ``--data-prop``. Raises ``RuntimeError`` with full diagnostics on failure.
    """
    model_path = Path(model_path)
    output_path = Path(output_path)

    # Pre-validation: verify ONNX model integrity
    try:
        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)
        if is_verbose():
            print(f"[MLIR] ONNX model validation passed: {model_path}")
    except Exception as e:
        raise RuntimeError(
            f"ONNX model validation failed for {model_path}\n"
            f"Error: {e}"
        )

    # Attempt ONNX to MLIR conversion with comprehensive error handling
    try:
        result = subprocess.run(
            [sys.executable, "-m", "iree.compiler.tools.import_onnx",
             str(model_path), "-o", str(output_path), "--data-prop"],
            capture_output=True, text=True, timeout=timeout
        )

        if result.returncode != 0:
            # Provide full diagnostic information
            error_msg = f"iree.compiler.tools.import_onnx failed for {model_path}\n"
            error_msg += f"Return code: {result.returncode}\n"
            error_msg += f"stdout:\n{result.stdout or '(empty)'}\n"
            error_msg += f"stderr:\n{result.stderr or '(empty)'}\n"
            error_msg += f"Model file size: {model_path.stat().st_size if model_path.exists() else 'N/A'} bytes"
            raise RuntimeError(error_msg)

        if is_verbose():
            print(f"[MLIR] Successfully converted {model_path} to {output_path}")

    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"iree.compiler.tools.import_onnx timed out for {model_path}\n"
            "This may indicate the model is too large or complex for the current environment."
        )
    except Exception as e:
        raise RuntimeError(
            f"iree.compiler.tools.import_onnx failed with exception for {model_path}\n"
            f"Error: {type(e).__name__}: {e}"
        )
