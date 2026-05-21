# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Utilities for locating and invoking Torq compiler tool binaries."""

import logging
import os
import platform
import subprocess
import sys
from typing import List, Optional

__all__ = [
    "find_tool",
    "invoke_immediate",
    "CompilerToolError",
]

_BUILTIN_TOOLS = [
    "torq-compile",
]

# Environment variable holding directories to search for named tools.
# Delimited by os.pathsep.
_TOOL_PATH_ENVVAR = "TORQ_TOOL_PATH"

logger = logging.getLogger(__name__)


class CompilerToolError(Exception):
    """Compiler exception that preserves the command line and error output."""

    def __init__(self, process: subprocess.CompletedProcess):
        try:
            errs = process.stderr.decode("utf-8")
        except Exception:
            errs = str(process.stderr)

        tool_name = os.path.basename(process.args[0])
        super().__init__(
            f"Error invoking Torq compiler tool {tool_name}\n"
            f"Error code: {process.returncode}\n"
            f"Diagnostics:\n{errs}\n\n"
            f"Invoked with:\n  {tool_name} {' '.join(process.args[1:])}\n\n"
            f"Ensure 'torq-compile' is installed and accessible, or set\n"
            f"the {_TOOL_PATH_ENVVAR} environment variable.\n"
        )


def find_tool(exe_name: str = "torq-compile") -> str:
    """Locate a Torq compiler tool binary.

    Search order:
      1. TORQ_TOOL_PATH environment variable directories
      2. Bundled binary alongside this package (_mlir_libs/ or _compiler_libs/)
      3. iree.compiler._mlir_libs package directory
      4. System PATH (via shutil.which)

    Args:
      exe_name: The tool executable name.

    Returns:
      Absolute path to the tool.

    Raises:
      FileNotFoundError: If the tool cannot be located.
    """
    if platform.system() == "Windows":
        exe_name_with_ext = exe_name + ".exe"
    else:
        exe_name_with_ext = exe_name

    # check TORQ_TOOL_PATH environment variable
    tool_path_env = os.environ.get(_TOOL_PATH_ENVVAR, "")
    if tool_path_env:
        for path_entry in tool_path_env.split(os.pathsep):
            candidate = os.path.join(path_entry, exe_name_with_ext)
            if _is_executable(candidate):
                return candidate

    # check bundled location torq/_compiler_libs/
    this_dir = os.path.dirname(__file__)
    bundled = os.path.join(this_dir, "..", "..", "_compiler_libs", exe_name_with_ext)
    if _is_executable(bundled):
        return os.path.realpath(bundled)

    # check iree/compiler/_mlir_libs/
    try:
        import iree.compiler._mlir_libs as _mlir_libs

        mlir_libs_dir = os.path.dirname(_mlir_libs.__file__)
        candidate = os.path.join(mlir_libs_dir, exe_name_with_ext)
        if _is_executable(candidate):
            return candidate
    except ImportError:
        pass

    # fall-back to system PATH
    import shutil

    system_path = shutil.which(exe_name)
    if system_path:
        return system_path

    raise FileNotFoundError(
        f"Torq compiler tool '{exe_name}' not found.\n"
        f"Set {_TOOL_PATH_ENVVAR} or ensure it is installed and on PATH."
    )


def invoke_immediate(
    command_line: List[str], *, immediate_input: Optional[bytes] = None
) -> bytes:
    """Invoke a compiler tool and return its stdout as bytes.

    Args:
      command_line: Full command line (tool path + arguments).
      immediate_input: Optional bytes to feed to stdin.

    Returns:
      The stdout of the process as bytes.

    Raises:
      CompilerToolError: If the process exits with non-zero status.
    """
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Invoke Torq tool: %s", " ".join(command_line))

    process = subprocess.run(
        command_line,
        input=immediate_input,
        capture_output=True,
    )
    if process.returncode != 0:
        raise CompilerToolError(process)
    if process.stderr:
        sys.stderr.buffer.write(process.stderr)
    return process.stdout


def _is_executable(path: str) -> bool:
    return bool(path and os.path.isfile(path) and os.access(path, os.X_OK))


def main(args=None):
    """Entry point for the torq-compile console script."""
    if args is None:
        args = sys.argv[1:]
    exe = find_tool("torq-compile")
    return subprocess.call(args=[exe] + args)
