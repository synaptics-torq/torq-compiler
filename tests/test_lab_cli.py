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

from torq.lab.cli import commands as cli
from _lab_fake_tools import write_fake_compile, write_fake_run

TOSA_MLIR = """
module {
  func.func @main(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
    return %arg0 : tensor<1x4xf32>
  }
}
"""

DYNAMIC_MLIR = """
module {
  func.func @main(%arg0: tensor<?x4xbf16>) -> tensor<?x4xbf16> {
    return %arg0 : tensor<?x4xbf16>
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


BF16_MLIR = """
module {
  func.func @main(%arg0: tensor<1x4xbf16>) -> tensor<1x4xbf16> {
    return %arg0 : tensor<1x4xbf16>
  }
}
"""

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


def test_convert_static_without_tensorflow_fails_cleanly(tmp_path, monkeypatch, capsys):
    """QA C2: without tensorflow the converter fails with a clean error naming
    the [tf] extra, not a raw ModuleNotFoundError traceback."""
    import sys

    from torq.lab.model_tools.shape_conversion import tflite as convert_static

    # Simulate an env without tensorflow: evict any cached tensorflow modules
    # (the torq.testing plugin imports TF at session start) and block the
    # parent so the lazy schema import raises ImportError.
    for name in [m for m in sys.modules if m == "tensorflow" or m.startswith("tensorflow.")]:
        monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "tensorflow", None)
    monkeypatch.setattr(convert_static, "_tflite_schema", None)

    rc = convert_static.main(["tflite", "-i", "model.tflite", "-o", "out.tflite"])
    err = capsys.readouterr().err
    assert rc == 1
    assert err.startswith("Error:")
    assert "tensorflow" in err and "[tf]" in err
    assert "Traceback" not in err


def test_cli_profile_without_extra_reports_raw_and_succeeds(fake_tools, monkeypatch, caplog):
    # with debug info present but the [profile] extra missing, profile
    # must report the raw host profile and exit 0 (the documented behavior),
    # not hard-fail.
    from torq.lab.profiling import annotate

    monkeypatch.setattr(annotate, "pd", None)
    caplog.set_level("INFO")
    model = _write_model(fake_tools)
    work = fake_tools / "profraw"
    (work / "debug").mkdir(parents=True)
    (work / "debug" / "info.json").write_text("{}")
    rc = cli.main(["profile", str(model), "--work-dir", str(work), "--random-inputs"])
    assert rc == 0
    assert "raw" in caplog.text
    assert "[profile] extra is not installed" in caplog.text


def test_cli_verify_bfloat16_golden_roundtrip(fake_tools, monkeypatch):
    # a bf16 .npy golden (saved via ml_dtypes, so it loads back as the
    # void |V2 dtype) must not crash verify with a raw TypeError; the fake run
    # emits zero-filled bf16 outputs, so a zero bf16 golden passes.
    import ml_dtypes
    import numpy as np

    monkeypatch.setenv("FAKE_OUTPUT_SIZES", str(1 * 4 * 2))
    model = fake_tools / "bf16.mlir"
    model.write_text(BF16_MLIR)
    work = fake_tools / "bf16out"
    golden = fake_tools / "g0.npy"
    np.save(golden, np.zeros((1, 4), dtype=ml_dtypes.bfloat16))
    rc = cli.main(["verify", str(model), "--work-dir", str(work), "--random-inputs", "--golden", str(golden)])
    assert rc == 0

    # An f32 golden against the bf16 output compares as usual (no crash).
    golden32 = fake_tools / "g0_f32.npy"
    np.save(golden32, np.zeros((1, 4), dtype=np.float32))
    rc = cli.main(["verify", str(model), "--work-dir", str(work), "--random-inputs", "--golden", str(golden32)])
    assert rc == 0


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
    from torq.lab.pipeline.workflow import ModelPipeline, PipelineConfig, RunResult

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


def test_cli_dynamic_shape_compile_fails_cleanly_without_invoking_the_compiler(fake_tools, caplog):
    # QA C3: a dynamic-shape source must fail before torq-compile is invoked,
    # with an actionable message instead of a raw compiler error.
    model = fake_tools / "dyn.mlir"
    model.write_text(DYNAMIC_MLIR)
    work = fake_tools / "dynout"
    rc = cli.main(["compile", str(model), "--work-dir", str(work)])
    assert rc == 1
    assert "does not support dynamic shapes" in caplog.text
    assert "Re-export the model with static shapes" in caplog.text
    assert not (fake_tools / "c_argv.txt").exists(), "torq-compile must not be invoked"


def test_cli_dynamic_shape_run_single_clean_error(fake_tools, caplog, capsys):
    # QA C3: `run` on a dynamic source must report one clean error (not two
    # stacked tracebacks); the error handler's manifest write re-parses the
    # same dynamic MLIR, which used to crash on the '?' dimension.
    model = fake_tools / "dyn.mlir"
    model.write_text(DYNAMIC_MLIR)
    work = fake_tools / "dynrun"
    rc = cli.main(["run", str(model), "--work-dir", str(work), "--random-inputs"])
    assert rc == 1
    assert "does not support dynamic shapes" in caplog.text
    out = capsys.readouterr().out
    err = capsys.readouterr().err
    assert "Traceback" not in out + err
    manifest = json.loads((work / "manifest.json").read_text())
    assert manifest["status"] == "error"
    assert "does not support dynamic shapes" in manifest["results"]["diagnostics"]


def test_cli_inspect_dynamic_shape_reports_question_dims(fake_tools, capsys):
    # QA C3: inspect on a dynamic-dim MLIR must not crash; '?' is shown.
    model = fake_tools / "dyn.mlir"
    model.write_text(DYNAMIC_MLIR)
    rc = cli.main(["inspect", str(model)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "?x4xbf16" in out


def test_cli_run_rejects_invalid_input_spec_dtype(fake_tools, caplog):
    # QA M1: an unknown dtype in --input-spec must be rejected, not silently
    # substituted with f32.
    model = _write_model(fake_tools)
    work = fake_tools / "specout"
    rc = cli.main(["compile", str(model), "--work-dir", str(work)])
    assert rc == 0
    vmfb = work / "model.vmfb"
    rc = cli.main(
        ["run", str(vmfb), "--work-dir", str(work), "--input-spec", "1x4xf99", "--random-inputs"]
    )
    assert rc == 1
    assert "unsupported dtype 'f99'" in caplog.text


def test_cli_internal_dynamic_error_maps_to_friendly_message(tmp_path, monkeypatch, caplog):
    # Dynamic dims hidden behind a static signature reach the backend as a
    # generic compiler error; the pipeline maps it to the same actionable
    # "dynamic shapes not supported" message.
    import torq.lab.pipeline.io as lab_io

    bad = tmp_path / "tools" / "torq-compile"
    bad.parent.mkdir(parents=True)
    bad.write_text(
        "#!/bin/sh\n"
        "echo 'error: \"unhandled dynamic dimensions for created dispatch region\"' >&2\n"
        "exit 1\n"
    )
    bad.chmod(0o755)
    monkeypatch.setenv("TORQ_COMPILE", str(bad))
    monkeypatch.setattr(lab_io, "dynamic_io_types", lambda *a, **k: [])

    model = _write_model(tmp_path)
    rc = cli.main(["compile", str(model), "--work-dir", str(tmp_path / "out")])
    assert rc == 1
    assert "does not support dynamic shapes" in caplog.text


def test_cli_inspect_function_override(fake_tools, capsys):
    model = _write_model(fake_tools)
    rc = cli.main(["inspect", str(model), "--function", "custom_fn"])
    assert rc == 0
    assert "Function: custom_fn [--function override]" in capsys.readouterr().out
