# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the torq.lab.profiling orchestration helpers and pipeline wiring.

The heavy annotation internals (pandas / perfetto) are faked so these run
without the optional ``[profile]`` extra and without real debug-info fixtures.
The end-to-end annotation of real debug info is exercised by the pytest suite
and the compile-run smoke test, not here.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from torq.lab import LabError
from torq.lab.pipeline.workflow import ModelPipeline, PipelineConfig
from torq.lab.profiling import annotate as profiling
from _lab_fake_tools import write_fake_compile, write_fake_run

TOSA_MLIR = """
module {
  func.func @main(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
    return %arg0 : tensor<1x4xf32>
  }
}
"""


# -- orchestration helpers (faked internals) -----------------------------


def test_perfetto_module_help_matches_documented_invocation():
    pytest.importorskip("google.protobuf")
    result = subprocess.run(
        [sys.executable, "-m", "torq.lab.profiling.perfetto", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "Convert timeline CSV logs to Perfetto trace" in result.stdout


def test_profiling_helpers_require_extra(tmp_path, monkeypatch):
    # Simulate the [profile] extra being absent: the guard trips before any work.
    monkeypatch.setattr(profiling, "pd", None)
    with pytest.raises(LabError):
        profiling.annotate_run_profile(tmp_path / "d", tmp_path / "h.csv", tmp_path / "p")
    with pytest.raises(LabError):
        profiling.write_compile_trace(tmp_path / "d", tmp_path / "p")
    with pytest.raises(LabError):
        profiling.write_perfetto_report([], tmp_path / "v.html")


# -- pipeline wiring (faked profiling helpers) ---------------------------


def test_compile_run_profile_runtime_wires_viewer(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_COMPILE_RECORD", str(tmp_path / "c.txt"))
    monkeypatch.setenv("FAKE_RUN_RECORD", str(tmp_path / "r.txt"))
    monkeypatch.setenv("FAKE_OUTPUT_SIZES", str(1 * 4 * 4))
    compile_tool = write_fake_compile(tmp_path / "tools")
    run_tool = write_fake_run(tmp_path / "tools")

    def fake_annotate(debug, host, out_dir):
        out_dir = Path(out_dir)
        (out_dir / "annotated_profile.xlsx").write_text("x")
        trace = out_dir / "trace.pb"
        trace.write_bytes(b"pb")
        return {"annotated": out_dir / "annotated_profile.xlsx", "trace": trace}

    def fake_report(pb_files, out_html):
        Path(out_html).write_text("<html></html>")
        return Path(out_html)

    monkeypatch.setattr(profiling, "annotate_run_profile", fake_annotate)
    monkeypatch.setattr(profiling, "write_perfetto_report", fake_report)

    model = tmp_path / "model.mlir"
    model.write_text(TOSA_MLIR)
    work = tmp_path / "out"
    # The fake compiler emits no debug info; seed a marker so the "has debug
    # info" gate lets annotation run (compile() keeps existing dir contents).
    (work / "debug").mkdir(parents=True, exist_ok=True)
    (work / "debug" / "info.json").write_text("{}")
    config = PipelineConfig(
        model_path=model, work_dir=work,
        compile_tool=str(compile_tool), run_tool=str(run_tool),
        random_inputs=True, profile_runtime=True,
    )
    pipe = ModelPipeline(config)
    compile_result, run_result = pipe.compile_run()

    profiles = work / "profiles"
    assert run_result.host_profile == profiles / "host_profile.csv"
    assert run_result.trace == profiles / "trace.pb"
    assert run_result.annotated_profile == profiles / "annotated_profile.xlsx"
    assert run_result.perfetto_viewer == profiles / "perfetto_viewer.html"
    assert run_result.perfetto_viewer.exists()

    manifest_path = pipe.write_manifest(compile_result=compile_result, run_result=run_result)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["results"]["run"]["perfetto_viewer"] == str(profiles / "perfetto_viewer.html")


def test_profile_sets_runtime_flag_before_compiling(tmp_path, monkeypatch):
    # profile() must set profile_runtime before ensure_vmfb() compiles a
    # source, so the compile picks up --torq-enable-profiling.
    monkeypatch.setenv("FAKE_COMPILE_RECORD", str(tmp_path / "c.txt"))
    monkeypatch.setenv("FAKE_RUN_RECORD", str(tmp_path / "r.txt"))
    monkeypatch.setenv("FAKE_OUTPUT_SIZES", str(1 * 4 * 4))
    compile_tool = write_fake_compile(tmp_path / "tools")
    run_tool = write_fake_run(tmp_path / "tools")

    def fake_annotate(debug, host, out_dir):
        out_dir = Path(out_dir)
        (out_dir / "annotated_profile.xlsx").write_text("x")
        trace = out_dir / "trace.pb"
        trace.write_bytes(b"pb")
        return {"annotated": out_dir / "annotated_profile.xlsx", "trace": trace}

    def fake_report(pb_files, out_html):
        Path(out_html).write_text("<html></html>")
        return Path(out_html)

    monkeypatch.setattr(profiling, "annotate_run_profile", fake_annotate)
    monkeypatch.setattr(profiling, "write_perfetto_report", fake_report)

    model = tmp_path / "model.mlir"
    model.write_text(TOSA_MLIR)
    work = tmp_path / "out"
    # The fake compiler emits no debug info; seed a marker so the "has debug
    # info" gate lets annotation run (compile() keeps existing dir contents).
    (work / "debug").mkdir(parents=True, exist_ok=True)
    (work / "debug" / "info.json").write_text("{}")
    config = PipelineConfig(
        model_path=model, work_dir=work,
        compile_tool=str(compile_tool), run_tool=str(run_tool),
        random_inputs=True,
    )
    pipe = ModelPipeline(config)
    compile_result, run_result = pipe.profile()

    assert config.profile_runtime is True
    assert "--torq-enable-profiling" in (tmp_path / "c.txt").read_text()
    assert run_result.annotated_profile == work / "profiles" / "annotated_profile.xlsx"
    assert run_result.perfetto_viewer == work / "profiles" / "perfetto_viewer.html"


def test_run_without_profiling_skips_finalize(tmp_path, monkeypatch):
    # No profiling flags -> _finalize_profiles must not import or call profiling.
    monkeypatch.setenv("FAKE_COMPILE_RECORD", str(tmp_path / "c.txt"))
    monkeypatch.setenv("FAKE_RUN_RECORD", str(tmp_path / "r.txt"))
    monkeypatch.setenv("FAKE_OUTPUT_SIZES", str(1 * 4 * 4))
    compile_tool = write_fake_compile(tmp_path / "tools")
    run_tool = write_fake_run(tmp_path / "tools")

    def boom(*a, **k):
        raise AssertionError("profiling must not run without a profiling flag")

    monkeypatch.setattr(profiling, "annotate_run_profile", boom)
    monkeypatch.setattr(profiling, "write_perfetto_report", boom)
    monkeypatch.setattr(profiling, "write_compile_trace", boom)

    model = tmp_path / "model.mlir"
    model.write_text(TOSA_MLIR)
    config = PipelineConfig(
        model_path=model, work_dir=tmp_path / "out",
        compile_tool=str(compile_tool), run_tool=str(run_tool),
        random_inputs=True,
    )
    _, run_result = ModelPipeline(config).compile_run()
    assert run_result.perfetto_viewer is None
