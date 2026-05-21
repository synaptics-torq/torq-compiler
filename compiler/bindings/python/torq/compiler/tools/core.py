# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Core compiler interface: CompilerOptions, compile_file, compile_str."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Optional, Sequence, Union

from .binaries import find_tool, invoke_immediate

__all__ = [
    "compile_file",
    "compile_str",
    "CompilerOptions",
]


@dataclass
class CompilerOptions:
    """Options for the Torq compiler.

    Arguments:
      output_file: Optionally save the compiled binary to a file instead of
        returning it.
      target_backends: List of target backend names to compile for.
        Common values: "torq", "llvm-cpu".
      input_type: The input type for legalization (e.g. "auto", "none", "tosa",
        "torch", "stablehlo", "onnx"). Defaults to "auto".
      extra_args: Additional arguments passed directly to torq-compile.
      crash_reproducer_path: File path to output an MLIR crash dump on failure.
    """

    output_file: Optional[str] = None
    target_backends: Sequence[str] = ("torq",)
    input_type: str = "auto"
    extra_args: Sequence[str] = ()
    crash_reproducer_path: Optional[str] = None


def _build_command_line(input_file: str, options: CompilerOptions) -> list:
    """Build the torq-compile command line from options."""
    tool = find_tool("torq-compile")
    cl = [tool, input_file]

    if options.input_type:
        cl.append(f"--iree-input-type={options.input_type}")

    cl.append("--iree-vm-bytecode-module-output-format=flatbuffer-binary")

    for backend in options.target_backends:
        cl.append(f"--iree-hal-target-backends={backend}")

    if options.output_file:
        cl.append(f"-o={options.output_file}")

    if options.crash_reproducer_path:
        cl.append(
            f"--mlir-pass-pipeline-crash-reproducer={options.crash_reproducer_path}"
        )

    cl.extend(options.extra_args)
    return cl


def compile_file(input_file: Union[str, os.PathLike], **kwargs) -> Optional[bytes]:
    """Compile an MLIR file using torq-compile.

    Args:
      input_file: Path to the MLIR file to compile.
      **kwargs: Keyword arguments corresponding to CompilerOptions fields.

    Returns:
      Compiled VMFB binary as bytes, or None if output_file was specified.
    """
    options = CompilerOptions(**kwargs)
    input_path = str(input_file)

    if options.output_file:
        cl = _build_command_line(input_path, options)
        invoke_immediate(cl)
        return None

    with tempfile.NamedTemporaryFile(suffix=".vmfb", delete=True) as tmp:
        options.output_file = tmp.name
        cl = _build_command_line(input_path, options)
        invoke_immediate(cl)
        with open(tmp.name, "rb") as f:
            return f.read()


def compile_str(input_str: Union[str, bytes], **kwargs) -> Optional[bytes]:
    """Compile MLIR source text using torq-compile.

    Args:
      input_str: MLIR assembly text (str or bytes) to compile.
      **kwargs: Keyword arguments corresponding to CompilerOptions fields.

    Returns:
      Compiled VMFB binary as bytes, or None if output_file was specified.
    """
    options = CompilerOptions(**kwargs)

    with tempfile.NamedTemporaryFile(suffix=".mlir", mode="wb", delete=True) as f:
        if isinstance(input_str, str):
            f.write(input_str.encode("utf-8"))
        else:
            f.write(input_str)
        f.flush()

        if options.output_file:
            cl = _build_command_line(f.name, options)
            invoke_immediate(cl)
            return None

        with tempfile.NamedTemporaryFile(suffix=".vmfb", delete=True) as out:
            options.output_file = out.name
            cl = _build_command_line(f.name, options)
            invoke_immediate(cl)
            with open(out.name, "rb") as result:
                return result.read()
