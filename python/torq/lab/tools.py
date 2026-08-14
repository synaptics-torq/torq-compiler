# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Discovery of the torq-compile / torq-run-module binaries and a subprocess wrapper."""

import logging
import os
import shutil
import subprocess
from typing import List, Optional, Sequence

from torq.lab.types import LabError

logger = logging.getLogger("torq.lab.tools")

# Extra directories to search, delimited by os.pathsep.
_TOOL_PATH_ENVVAR = "TORQ_TOOL_PATH"


class ToolError(LabError):
    """A tool invocation failed; preserves the command line and diagnostics."""

    def __init__(self, cmd: Sequence[str], returncode: Optional[int], output):
        self.cmd = list(cmd)
        self.returncode = returncode
        if isinstance(output, bytes):
            output = output.decode("utf-8", errors="replace")
        self.output = output or ""
        tool_name = os.path.basename(str(self.cmd[0])) if self.cmd else "tool"
        code = "timeout" if returncode is None else f"code {returncode}"
        super().__init__(
            f"Error invoking {tool_name} ({code})\n"
            f"Diagnostics:\n{self.output}\n\n"
            f"Command:\n  {' '.join(str(c) for c in self.cmd)}\n"
        )


def _is_executable(path) -> bool:
    return bool(path and os.path.isfile(path) and os.access(path, os.X_OK))


def _compiler_packaged_dirs() -> List[str]:
    dirs: List[str] = []
    try:
        import iree.compiler._mlir_libs as _mlir_libs

        dirs.append(os.path.dirname(_mlir_libs.__file__))
    except ImportError:
        pass
    try:
        import torq._compiler_libs as _compiler_libs

        dirs.append(os.path.dirname(_compiler_libs.__file__))
    except ImportError:
        pass
    return dirs


def _runtime_packaged_dirs() -> List[str]:
    dirs: List[str] = []
    try:
        import torq._runtime_libs as _runtime_libs

        dirs.append(os.path.dirname(_runtime_libs.__file__))
    except ImportError:
        pass
    return dirs


def _find_tool(exe_name: str, explicit, env_var: str, packaged_dirs) -> str:
    """Locate a tool binary using the torq.lab discovery precedence.

    Order: explicit path, then env var (direct path and TORQ_TOOL_PATH dirs),
    then packaged compiler/runtime wrappers, then PATH.
    """
    # 1. explicit config path
    if explicit:
        if _is_executable(explicit):
            return str(explicit)
        which = shutil.which(str(explicit))
        if which:
            return which
        raise FileNotFoundError(f"Configured {exe_name} path is not executable: {explicit}")

    # 2a. env var pointing directly at the binary
    env_path = os.getenv(env_var)
    if env_path and _is_executable(env_path):
        return env_path

    # 2b. TORQ_TOOL_PATH directories
    for path_entry in os.getenv(_TOOL_PATH_ENVVAR, "").split(os.pathsep):
        candidate = os.path.join(path_entry, exe_name)
        if _is_executable(candidate):
            return candidate

    # 3. packaged compiler/runtime wrappers
    for directory in packaged_dirs():
        candidate = os.path.join(directory, exe_name)
        if _is_executable(candidate):
            return candidate

    # 4. system PATH
    system_path = shutil.which(exe_name)
    if system_path:
        return system_path

    raise FileNotFoundError(
        f"Torq tool '{exe_name}' not found. Set {env_var} or {_TOOL_PATH_ENVVAR}, "
        f"install the torq-compiler/torq-runtime wheels, or put it on PATH."
    )


def find_compile_tool(explicit=None) -> str:
    """Locate the ``torq-compile`` binary."""
    return _find_tool("torq-compile", explicit, "TORQ_COMPILE", _compiler_packaged_dirs)


def find_run_tool(explicit=None) -> str:
    """Locate the ``torq-run-module`` binary."""
    return _find_tool("torq-run-module", explicit, "TORQ_RUN_MODULE", _runtime_packaged_dirs)


def run_tool(cmd: Sequence[str], *, timeout=None, cwd=None) -> subprocess.CompletedProcess:
    """Run a tool command, raising ToolError on non-zero exit or timeout."""
    cmd = [str(c) for c in cmd]
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("Invoke: %s", " ".join(cmd))
    try:
        proc = subprocess.run(cmd, capture_output=True, cwd=cwd, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise ToolError(cmd, None, exc.stderr or f"timed out after {timeout}s") from exc
    if proc.returncode != 0:
        raise ToolError(cmd, proc.returncode, proc.stderr)
    return proc
