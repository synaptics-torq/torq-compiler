# Copyright 2025-2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""TFLite-to-MLIR model import.

Source-model importing for the pipeline: converts a TFLite model to MLIR
via ``tosa-converter-for-tflite``. FlatBuffer layer extraction lives in
:mod:`torq.lab.model_tools.extraction.tflite`.
"""

import shutil
import subprocess
from pathlib import Path

from torq.lab.logging import is_verbose

def convert_tflite_to_mlir(model_path, output_path, timeout=300):
    """Convert a TFLite model file to MLIR via ``tosa-converter-for-tflite``.

    Runs the converter in a subprocess with ``--text``. Raises ``RuntimeError``
    with full diagnostics on failure.
    """
    model_path = Path(model_path)
    output_path = Path(output_path)

    tool = shutil.which("tosa-converter-for-tflite")
    if tool is None:
        raise RuntimeError(
            "tosa-converter-for-tflite not found on PATH; install the [tflite] extra"
        )

    try:
        result = subprocess.run(
            [tool, str(model_path), "--text", "-o", str(output_path)],
            capture_output=True, text=True, timeout=timeout
        )

        if result.returncode != 0:
            # Provide full diagnostic information
            error_msg = f"tosa-converter-for-tflite failed for {model_path}\n"
            error_msg += f"Return code: {result.returncode}\n"
            error_msg += f"stdout:\n{result.stdout or '(empty)'}\n"
            error_msg += f"stderr:\n{result.stderr or '(empty)'}\n"
            error_msg += f"Model file size: {model_path.stat().st_size if model_path.exists() else 'N/A'} bytes"
            raise RuntimeError(error_msg)

        if is_verbose():
            print(f"[MLIR] Successfully converted {model_path} to {output_path}")

    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"tosa-converter-for-tflite timed out for {model_path}\n"
            "This may indicate the model is too large or complex for the current environment."
        )
    except Exception as e:
        raise RuntimeError(
            f"tosa-converter-for-tflite failed with exception for {model_path}\n"
            f"Error: {type(e).__name__}: {e}"
        )
