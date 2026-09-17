# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Discovery of the torq-compile / torq-run-module binaries and a subprocess wrapper."""

import importlib.util
import logging
import os
import shutil
import subprocess
from typing import List, Optional, Sequence

from torq.lab import LabError

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


def _module_dir(module_name: str) -> Optional[str]:
    """Resolve a module's on-disk directory without importing it.

    Locating a packaged tool's directory only needs the module's filesystem
    location, not its code loaded into this process. Some of these modules
    (notably ``iree.compiler._mlir_libs``) are native extensions that
    statically link their own copy of LLVM; importing one into a process that
    already has another LLVM-linked extension loaded (e.g. TensorFlow, pulled
    in for TFLite import) causes an LLVM commandline registration error. 
    ``find_spec`` locates the module via its parent package's finders without 
    executing the module itself, avoiding that collision entirely.
    """
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ValueError, AttributeError):
        return None
    if spec is None:
        return None
    if spec.submodule_search_locations:
        return spec.submodule_search_locations[0]
    if spec.origin:
        return os.path.dirname(spec.origin)
    return None


def _compiler_packaged_dirs() -> List[str]:
    return [
        d for d in (_module_dir(name) for name in ("iree.compiler._mlir_libs", "torq._compiler_libs")) if d
    ]


def _runtime_packaged_dirs() -> List[str]:
    return [d for d in (_module_dir("torq._runtime_libs"),) if d]


def _iree_runtime_packaged_dirs() -> List[str]:
    dirs: List[str] = _runtime_packaged_dirs()
    # The iree-runtime wheel ships iree-run-module alongside its native libs.
    for module_name in ("iree._runtime_libs", "iree.runtime._runtime_libs"):
        directory = _module_dir(module_name)
        if directory:
            dirs.append(directory)
    return dirs


def _dev_tree_tool_dirs() -> List[str]:
    """Tool directories of a sibling IREE build tree (source-checkout layout).

    Mirrors the legacy ``testing/iree.py`` BUILD_DIR fallback so the lab works
    from a development checkout without any environment variables: honor
    ``IREE_BUILD_DIR`` when set, otherwise walk up from each ``torq`` package
    portion looking for a build tree (``.../bindings/python/torq``) or for a
    source checkout with a sibling ``iree-build`` (``<repo>/python/torq``).
    Returns nothing for wheel installs, where no such build tree exists.
    """
    try:
        import torq

        path_entries = list(getattr(torq, "__path__", []) or [])
        torq_file = getattr(torq, "__file__", None)
        if torq_file:
            path_entries.append(os.path.dirname(torq_file))
    except Exception:
        return []
    tool_subdirs = ("third_party/iree/tools", "runtime/tools", "compiler/tools")
    build_roots: List[str] = []
    env_build_dir = os.getenv("IREE_BUILD_DIR")
    if env_build_dir:
        build_roots.append(env_build_dir)
    for entry in path_entries:
        current = os.path.abspath(entry)
        for _ in range(8):
            current = os.path.dirname(current)
            candidates = [current, os.path.join(current, "iree-build")]
            if any(
                os.path.isdir(os.path.join(base, sub))
                for base in candidates
                for sub in tool_subdirs
            ):
                build_roots.extend(candidates)
                break
    dirs: List[str] = []
    for base in build_roots:
        for sub in tool_subdirs:
            path = os.path.join(base, sub)
            if os.path.isdir(path) and path not in dirs:
                dirs.append(path)
    return dirs


def _find_tool(exe_name: str, explicit, env_var: str, packaged_dirs) -> str:
    """Locate a tool binary using the torq.lab discovery precedence.

    Order: explicit path, then env var (direct path and TORQ_TOOL_PATH dirs),
    then packaged compiler/runtime wrappers, then a sibling build tree of a
    source checkout (IREE_BUILD_DIR or ../iree-build), then PATH.
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

    # 3b. sibling build tree of a source checkout (dev environment)
    for directory in _dev_tree_tool_dirs():
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


def find_iree_compile_tool(explicit=None) -> str:
    """Locate the ``iree-compile`` binary (generic IREE compiler, e.g. for llvm-cpu)."""
    return _find_tool("iree-compile", explicit, "IREE_COMPILE", _compiler_packaged_dirs)


def find_iree_run_tool(explicit=None) -> str:
    """Locate the ``iree-run-module`` binary (generic IREE runtime)."""
    return _find_tool("iree-run-module", explicit, "IREE_RUN_MODULE", _iree_runtime_packaged_dirs)


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
