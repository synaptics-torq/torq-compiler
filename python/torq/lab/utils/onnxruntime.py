# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2026 Synaptics Incorporated.

"""CPU-only ONNX Runtime helpers."""

from __future__ import annotations

import os
import sys
from typing import Union


def _import_ort_silently():
    """Import ``onnxruntime`` with the native GPU-discovery warning silenced."""
    if "onnxruntime" in sys.modules:
        return sys.modules["onnxruntime"]

    try:
        stderr_fd = sys.stderr.fileno()
    except (AttributeError, OSError):
        # Stderr is not a real fd (e.g. captured by a test runner);
        # fall back to a plain import. Any warning will be captured anyway.
        import onnxruntime  # noqa: PLC0415

        return onnxruntime

    saved_fd = os.dup(stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull_fd, stderr_fd)
        import onnxruntime  # noqa: PLC0415
    finally:
        os.dup2(saved_fd, stderr_fd)
        os.close(saved_fd)
        os.close(devnull_fd)
    return onnxruntime


ort = _import_ort_silently()

# 0 = verbose, 1 = info, 2 = warning, 3 = error, 4 = fatal.
ort.set_default_logger_severity(3)

CPU_PROVIDERS: tuple[str, ...] = ("CPUExecutionProvider",)

ModelInput = Union[str, bytes, bytearray]


def make_cpu_session(
    model: ModelInput,
    *,
    sess_options: "ort.SessionOptions | None" = None,
) -> "ort.InferenceSession":
    """Return a CPU-only ONNX Runtime inference session with quiet logging."""
    opts = sess_options if sess_options is not None else ort.SessionOptions()
    opts.log_severity_level = 3
    return ort.InferenceSession(
        model,
        sess_options=opts,
        providers=list(CPU_PROVIDERS),
    )
