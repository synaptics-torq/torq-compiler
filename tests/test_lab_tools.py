# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unit tests for torq.lab.tools discovery precedence and the subprocess wrapper."""

import os
import stat

import pytest

from torq.lab import tools
from torq.lab.tools import ToolError


def _make_exe(directory, name="torq-compile", body="#!/bin/sh\nexit 0\n"):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return path


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for var in ("TORQ_COMPILE", "TORQ_RUN_MODULE", "TORQ_TOOL_PATH"):
        monkeypatch.delenv(var, raising=False)


def test_explicit_path_wins(tmp_path, monkeypatch):
    explicit = _make_exe(tmp_path / "explicit")
    env = _make_exe(tmp_path / "env")
    monkeypatch.setenv("TORQ_COMPILE", str(env))
    assert tools.find_compile_tool(explicit=str(explicit)) == str(explicit)


def test_explicit_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        tools.find_compile_tool(explicit=str(tmp_path / "nope"))


def test_env_var_beats_tool_path(tmp_path, monkeypatch):
    env = _make_exe(tmp_path / "env")
    toolpath = _make_exe(tmp_path / "toolpath")
    monkeypatch.setenv("TORQ_COMPILE", str(env))
    monkeypatch.setenv("TORQ_TOOL_PATH", str(toolpath.parent))
    assert tools.find_compile_tool() == str(env)


def test_tool_path_dir(tmp_path, monkeypatch):
    toolpath = _make_exe(tmp_path / "toolpath")
    monkeypatch.setenv("TORQ_TOOL_PATH", str(toolpath.parent))
    monkeypatch.setattr(tools, "_compiler_packaged_dirs", list)
    assert tools.find_compile_tool() == str(toolpath)


def test_packaged_dir_before_path(tmp_path, monkeypatch):
    packaged = _make_exe(tmp_path / "packaged")
    monkeypatch.setattr(tools, "_compiler_packaged_dirs", lambda: [str(packaged.parent)])
    assert tools.find_compile_tool() == str(packaged)


def test_path_fallback(tmp_path, monkeypatch):
    onpath = _make_exe(tmp_path / "bin", name="torq-run-module")
    monkeypatch.setenv("PATH", str(onpath.parent) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setattr(tools, "_runtime_packaged_dirs", list)
    monkeypatch.setattr(tools, "_dev_tree_tool_dirs", list)
    assert tools.find_run_tool() == str(onpath)


def test_dev_tree_before_path(tmp_path, monkeypatch):
    devtree = _make_exe(tmp_path / "build" / "runtime" / "tools")
    onpath = _make_exe(tmp_path / "bin")
    monkeypatch.setenv("PATH", str(onpath.parent) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setattr(tools, "_compiler_packaged_dirs", list)
    monkeypatch.setattr(tools, "_dev_tree_tool_dirs", lambda: [str(devtree.parent)])
    assert tools.find_compile_tool() == str(devtree)


def test_not_found(monkeypatch):
    monkeypatch.setattr(tools, "_compiler_packaged_dirs", list)
    monkeypatch.setattr(tools, "_dev_tree_tool_dirs", list)
    monkeypatch.setenv("PATH", "")
    with pytest.raises(FileNotFoundError):
        tools.find_compile_tool()


def test_run_tool_success(tmp_path):
    script = _make_exe(tmp_path, name="ok", body="#!/bin/sh\necho hello\n")
    proc = tools.run_tool([str(script)])
    assert proc.stdout.strip() == b"hello"


def test_run_tool_failure_raises(tmp_path):
    script = _make_exe(tmp_path, name="bad", body="#!/bin/sh\necho boom >&2\nexit 3\n")
    with pytest.raises(ToolError) as excinfo:
        tools.run_tool([str(script), "--flag"])
    err = excinfo.value
    assert err.returncode == 3
    assert "boom" in err.output
    assert "--flag" in str(err)


def test_run_tool_timeout_raises(tmp_path):
    script = _make_exe(tmp_path, name="slow", body="#!/bin/sh\nsleep 5\n")
    with pytest.raises(ToolError) as excinfo:
        tools.run_tool([str(script)], timeout=0.2)
    assert excinfo.value.returncode is None
