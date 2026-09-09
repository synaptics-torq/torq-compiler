# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CLI exit-code and guard contracts (hardware-free).

The happy-path compile/run/manifest plumbing is covered by real usage of the
pipeline; this file keeps the CLI-specific contracts a green end-to-end run
would not assert: the verify guard, expected-outputs coming from a config, and
the error exit code / manifest diagnostics on a tool failure.
"""

import json
from pathlib import Path

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


def test_cli_verify_passes_on_match(fake_tools):
    import numpy as np

    model = _write_model(fake_tools)
    work = fake_tools / "out"
    golden = fake_tools / "g0.npy"
    np.save(golden, np.zeros((1, 4), np.float32))
    rc = cli.main(["verify", str(model), "--work-dir", str(work), "--random-inputs", "--golden", str(golden)])
    assert rc == 0
    manifest = json.loads((work / "manifest.json").read_text())
    assert manifest["results"]["comparison"]["passed"] is True


def test_cli_verify_fails_on_mismatch(fake_tools):
    import numpy as np

    model = _write_model(fake_tools)
    work = fake_tools / "out"
    golden = fake_tools / "g0.npy"
    np.save(golden, np.ones((1, 4), np.float32))
    rc = cli.main(["verify", str(model), "--work-dir", str(work), "--random-inputs", "--golden", str(golden)])
    assert rc == 2


def test_cli_verify_swallowed_positional_reports_fix(fake_tools, caplog):
    model = _write_model(fake_tools)
    rc = cli.main(["verify", "--golden", "a.npy", "b.npy", str(model)])
    assert rc == 1
    assert f"torq-lab verify {model} --golden a.npy b.npy" in caplog.text


def test_cli_verify_vmfb_without_golden_fails(fake_tools, caplog):
    vmfb = fake_tools / "model.vmfb"
    vmfb.write_bytes(b"FAKEVMFB")
    rc = cli.main(["verify", str(vmfb)])
    assert rc == 1
    assert "--golden" in caplog.text and "--reference" in caplog.text


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


def test_cli_run_uses_the_named_vmfb(fake_tools):
    art = fake_tools / "art"
    art.mkdir()
    (art / "qkv.mlir").write_text(TOSA_MLIR)
    (art / "qkv.vmfb").write_bytes(b"fake")
    work = fake_tools / "w"
    work.mkdir()
    (work / "model.vmfb").write_bytes(b"stale")   # must not shadow the positional
    rc = cli.main(["run", str(art / "qkv.vmfb"), "--work-dir", str(work), "--random-inputs"])
    assert rc == 0
    argv = (fake_tools / "r_argv.txt").read_text()
    assert f"--module={art / 'qkv.vmfb'}" in argv
    # A .vmfb positional must not invoke the compiler.
    assert not (fake_tools / "c_argv.txt").exists()


def test_cli_profile_enables_profiling_and_reports_raw(fake_tools, caplog):
    caplog.set_level("INFO")
    model = _write_model(fake_tools)
    work = fake_tools / "out"
    rc = cli.main(["profile", str(model), "--work-dir", str(work), "--random-inputs"])
    assert rc == 0
    assert "--torq-enable-profiling" in (fake_tools / "c_argv.txt").read_text()
    assert "--torq_profile_host=" in (fake_tools / "r_argv.txt").read_text()
    assert "raw" in caplog.text


def test_profile_quality_classifies_annotated_raw_and_neither(tmp_path):
    from torq.lab.pipeline import ModelPipeline
    from torq.lab.types import PipelineConfig, RunResult

    pipe = ModelPipeline(PipelineConfig(model_path=tmp_path / "m.mlir", work_dir=tmp_path / "w"))

    annotated = RunResult(command=[], annotated_profile=tmp_path / "a.xlsx", perfetto_viewer=tmp_path / "v.html")
    assert cli._profile_quality(pipe, annotated) == ("annotated", str(tmp_path / "v.html"))

    raw = RunResult(command=[], host_profile=tmp_path / "h.csv")
    assert cli._profile_quality(pipe, raw) == ("raw", "no debug info to annotate against")

    neither = RunResult(command=[])
    assert cli._profile_quality(pipe, neither) == (None, None)


_STAGE_FLAGS = {
    "compile":     (["--chip", "--dump-ir", "--profile-compile", "--output", "--print-plan", "--json"],
                    ["--remote", "--input-npy", "--random-inputs", "--golden", "--profile-runtime", "--input-spec", "--output-spec", "--seed"]),
    "verify":      (["--chip", "--input-npy", "--golden", "--remote", "--print-plan", "--json", "--input-spec", "--output-spec", "--seed"], ["--output ", "--profile-runtime"]),
}


@pytest.mark.parametrize("command", sorted(_STAGE_FLAGS))
def test_cli_help_is_stage_scoped(command, capsys):
    present, absent = _STAGE_FLAGS[command]
    with pytest.raises(SystemExit) as excinfo:
        cli.main([command, "--help"])
    assert excinfo.value.code == 0
    out = capsys.readouterr().out
    for flag in present:
        assert flag in out
    for flag in absent:
        assert flag not in out


@pytest.mark.parametrize("argv", [
    ["compile", "--remote", "board", "m.mlir"],
    ["run", "--output", "out.vmfb", "m.vmfb"],
])
def test_cli_rejects_foreign_stage_flags(argv):
    with pytest.raises(SystemExit) as excinfo:
        cli.main(argv)
    assert excinfo.value.code == 2


def test_cli_print_plan_performs_no_side_effects(fake_tools):
    model = _write_model(fake_tools)
    work = fake_tools / "out"
    rc = cli.main(["run", str(model), "--work-dir", str(work), "--random-inputs", "--print-plan"])
    assert rc == 0
    assert not (fake_tools / "c_argv.txt").exists()
    assert not (fake_tools / "r_argv.txt").exists()
    assert not (work / "manifest.json").exists()


def test_cli_run_json_reports_structured_summary(fake_tools, capsys):
    model = _write_model(fake_tools)
    work = fake_tools / "out"
    rc = cli.main(["run", str(model), "--work-dir", str(work), "--random-inputs", "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["command"] == "run"
    assert payload["status"] == "ok"
    assert {s["name"] for s in payload["stages"]} == {"compile", "run"}


def test_cli_inspect_function_override(fake_tools, capsys):
    model = _write_model(fake_tools)
    rc = cli.main(["inspect", str(model), "--function", "custom_fn"])
    assert rc == 0
    assert "Function: custom_fn [--function override]" in capsys.readouterr().out
