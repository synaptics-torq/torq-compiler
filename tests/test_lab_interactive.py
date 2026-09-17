# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""No-argument interactive mode (hardware-free): drives ``run_interactive`` with a
scripted reader/writer against the fake compile/run tools.
"""

import pytest

from torq.lab.cli import interactive
from _lab_fake_tools import write_fake_compile, write_fake_run

TOSA_MLIR = """
module {
  func.func @main(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
    return %arg0 : tensor<1x4xf32>
  }
}
"""


@pytest.fixture
def fake_tools(tmp_path, monkeypatch):
    compile_tool = write_fake_compile(tmp_path / "tools")
    run_tool = write_fake_run(tmp_path / "tools")
    monkeypatch.setenv("TORQ_COMPILE", str(compile_tool))
    monkeypatch.setenv("TORQ_RUN_MODULE", str(run_tool))
    monkeypatch.setenv("FAKE_COMPILE_RECORD", str(tmp_path / "c_argv.txt"))
    monkeypatch.setenv("FAKE_RUN_RECORD", str(tmp_path / "r_argv.txt"))
    monkeypatch.setenv("FAKE_OUTPUT_SIZES", str(1 * 4 * 4))  # [1,4] f32
    return tmp_path


def _write_model(directory):
    model = directory / "model.mlir"
    model.write_text(TOSA_MLIR)
    return model


def _scripted_reader(responses):
    it = iter(responses)

    def reader(prompt=""):
        try:
            return next(it)
        except StopIteration:
            raise EOFError()

    return reader


def _capturing_writer():
    lines = []

    def writer(*args):
        lines.append(" ".join(str(a) for a in args) if args else "")

    return lines, writer


def test_interactive_profile_first_generates_random_inputs(fake_tools):
    # Regression: choosing "profile" before ever choosing "run" must still
    # generate inputs for a model that requires them, instead of silently
    # invoking torq-run-module with zero --input args.
    model = _write_model(fake_tools)
    work = fake_tools / "out"
    reader = _scripted_reader([
        str(model), str(work), "", "", "", "",
        "profile", "",
        "quit",
    ])
    lines, writer = _capturing_writer()

    rc = interactive.run_interactive(reader=reader, writer=writer)

    assert rc == 0
    assert not any("Profile failed" in line for line in lines)
    argv = (fake_tools / "r_argv.txt").read_text()
    assert "--input=" in argv
