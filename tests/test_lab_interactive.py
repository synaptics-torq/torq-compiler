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


def test_interactive_compile_failure_prints_diagnosis_then_menu(tmp_path, monkeypatch):
    """A failing initial compile prints the diagnosis; the session continues."""
    compile_tool = write_fake_compile(tmp_path / "tools")
    run_tool = write_fake_run(tmp_path / "tools")
    monkeypatch.setenv("TORQ_COMPILE", str(compile_tool))
    monkeypatch.setenv("TORQ_RUN_MODULE", str(run_tool))
    monkeypatch.setenv("FAKE_RC", "1")
    monkeypatch.setenv("FAKE_STDERR", "model.mlir:1:1: error: something nobody has seen before\n")
    model = _write_model(tmp_path)
    work = tmp_path / "out"
    responses = iter([
        str(model), str(work), "", "", "", "",
        str(model), str(work), "", "", "", "",
        "quit",
    ])
    seen_model_prompt = [False]

    def reader(prompt=""):
        try:
            resp = next(responses)
        except StopIteration:
            raise EOFError()
        # The model prompt appears a second time only after the first
        # compile failed: clear the failure knobs before the retry.
        if prompt.startswith("Model file"):
            if seen_model_prompt[0]:
                monkeypatch.setenv("FAKE_RC", "0")
                monkeypatch.delenv("FAKE_STDERR")
            else:
                seen_model_prompt[0] = True
        return resp

    lines, writer = _capturing_writer()
    rc = interactive.run_interactive(reader=reader, writer=writer)

    assert rc == 0
    joined = "\n".join(lines)
    # The failure printed the diagnosis before returning to the prompt.
    assert "Compile failed:" in joined
    assert "diagnosis: unknown" in joined
    # The session continues: the retry compiles and the menu is reachable.
    assert joined.index("diagnosis: unknown") < joined.index("Compile finished.")
    assert "Menu:" in joined
