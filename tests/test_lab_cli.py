# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CLI exit-code and guard contracts (hardware-free).

The happy-path compile/run/manifest plumbing is covered by real usage of the
pipeline; this file keeps the CLI-specific contracts a green end-to-end run
would not assert: the compare guard, expected-outputs coming from a config, and
the error exit code / manifest diagnostics on a tool failure.
"""

import json

import pytest

from torq.lab import cli
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


def test_cli_compare_requires_expected_outputs(fake_tools):
    model = _write_model(fake_tools)
    rc = cli.main(["compare", str(model), "--work-dir", str(fake_tools / "out"), "--random-inputs"])
    assert rc == 1


def test_cli_compare_expected_from_config(fake_tools):
    import numpy as np

    model = _write_model(fake_tools)
    work = fake_tools / "out"
    expected = fake_tools / "exp0.npy"
    np.save(expected, np.zeros((1, 4), np.float32))
    cfg = fake_tools / "cfg.json"
    cfg.write_text(json.dumps({
        "model_path": str(model),
        "work_dir": str(work),
        "random_inputs": True,
        "expected_output_npy": [str(expected)],
    }))
    # Expected outputs supplied via --config must satisfy the compare guard.
    rc = cli.main(["compare", "--config", str(cfg)])
    assert rc == 0
    manifest = json.loads((work / "manifest.json").read_text())
    assert manifest["results"]["comparison"]["passed"] is True


def test_cli_compile_failure_returns_error(tmp_path, monkeypatch):
    # A fake compile tool that always fails.
    bad = tmp_path / "tools" / "torq-compile"
    bad.parent.mkdir(parents=True)
    bad.write_text("#!/bin/sh\necho oops >&2\nexit 1\n")
    bad.chmod(0o755)
    monkeypatch.setenv("TORQ_COMPILE", str(bad))
    model = _write_model(tmp_path)
    work = tmp_path / "out"
    rc = cli.main(["compile", str(model), "--work-dir", str(work)])
    assert rc == 1
    manifest = json.loads((work / "manifest.json").read_text())
    assert manifest["status"] == "error"
    assert "oops" in manifest["results"]["diagnostics"]


@pytest.mark.parametrize("argv", [["--help"], ["compile", "--help"], ["compile-run", "--help"]])
def test_cli_help(argv):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    assert excinfo.value.code == 0
