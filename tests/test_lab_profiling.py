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
from pathlib import Path

import pytest

from torq.lab import profiling
from torq.lab.pipeline import ModelPipeline
from torq.lab.types import LabError, PipelineConfig
from _lab_fake_tools import write_fake_compile, write_fake_run

TOSA_MLIR = """
module {
  func.func @main(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
    return %arg0 : tensor<1x4xf32>
  }
}
"""


# -- orchestration helpers (faked internals) -----------------------------


def test_annotate_run_profile_writes_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(profiling, "_require_profile_deps", lambda: None)
    seen = {}

    def fake_annotate(debug, host, outs):
        seen["outs"] = outs
        for o in outs:
            Path(o).write_text("x")
        return {"m": 1}

    monkeypatch.setattr(profiling, "annotate_host_profile_from_files", fake_annotate)

    out = tmp_path / "profiles"
    res = profiling.annotate_run_profile(tmp_path / "debug", tmp_path / "host_profile.csv", out)

    assert res["annotated"] == out / "annotated_profile.xlsx"
    assert res["trace"] == out / "trace.pb"
    assert res["annotated"].exists() and res["trace"].exists()
    # The underlying annotator is asked for both output files, as strings.
    assert seen["outs"] == [str(out / "annotated_profile.xlsx"), str(out / "trace.pb")]


def test_write_compile_trace_suffixes_and_cleans(tmp_path, monkeypatch):
    monkeypatch.setattr(profiling, "_require_profile_deps", lambda: None)

    class FakePerfettoLogger:
        @staticmethod
        def convert_to_perfetto(debug, staging):
            s = Path(staging)
            s.mkdir(parents=True, exist_ok=True)
            (s / "dispatch_a.pb").write_bytes(b"A")
            (s / "dispatch_b.pb").write_bytes(b"B")
            return {}

    monkeypatch.setattr(profiling, "perfetto_logger", FakePerfettoLogger, raising=False)

    out = tmp_path / "profiles"
    produced = profiling.write_compile_trace(tmp_path / "debug", out)

    assert sorted(p.name for p in produced) == ["dispatch_a_compile.pb", "dispatch_b_compile.pb"]
    assert all(p.exists() for p in produced)
    # staging directory is removed
    assert not (out / "_compile_trace").exists()


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


def test_compile_profile_compile_wires_viewer(tmp_path, monkeypatch):
    # A standalone `compile --profile-compile` must produce the compile trace
    # and viewer without needing a run.
    monkeypatch.setenv("FAKE_COMPILE_RECORD", str(tmp_path / "c.txt"))
    compile_tool = write_fake_compile(tmp_path / "tools")

    def fake_compile_trace(debug, out_dir):
        pb = Path(out_dir) / "dispatch_compile.pb"
        pb.write_bytes(b"pb")
        return [pb]

    def fake_report(pb_files, out_html):
        Path(out_html).write_text("<html></html>")
        return Path(out_html)

    monkeypatch.setattr(profiling, "write_compile_trace", fake_compile_trace)
    monkeypatch.setattr(profiling, "write_perfetto_report", fake_report)

    model = tmp_path / "model.mlir"
    model.write_text(TOSA_MLIR)
    work = tmp_path / "out"
    # The fake compiler emits no debug info; seed a marker so the "has debug
    # info" gate lets the compile trace run (compile() keeps existing contents).
    (work / "debug").mkdir(parents=True, exist_ok=True)
    (work / "debug" / "info.json").write_text("{}")
    config = PipelineConfig(
        model_path=model, work_dir=work,
        compile_tool=str(compile_tool), profile_compile=True,
    )
    pipe = ModelPipeline(config)
    compile_result = pipe.compile()

    profiles = work / "profiles"
    assert compile_result.compile_profile == profiles / "compile_profile.csv"
    assert compile_result.compile_trace == profiles / "dispatch_compile.pb"
    assert compile_result.perfetto_viewer == profiles / "perfetto_viewer.html"
    assert compile_result.perfetto_viewer.exists()

    manifest_path = pipe.write_manifest(compile_result=compile_result)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["results"]["compile"]["compile_trace"] == str(profiles / "dispatch_compile.pb")
    assert manifest["results"]["compile"]["perfetto_viewer"] == str(profiles / "perfetto_viewer.html")


def test_compile_without_profiling_skips_finalize(tmp_path, monkeypatch):
    # No profiling flag -> compile() must not call the profiling helpers.
    monkeypatch.setenv("FAKE_COMPILE_RECORD", str(tmp_path / "c.txt"))
    compile_tool = write_fake_compile(tmp_path / "tools")

    def boom(*a, **k):
        raise AssertionError("profiling must not run without a profiling flag")

    monkeypatch.setattr(profiling, "write_compile_trace", boom)
    monkeypatch.setattr(profiling, "write_perfetto_report", boom)

    model = tmp_path / "model.mlir"
    model.write_text(TOSA_MLIR)
    config = PipelineConfig(
        model_path=model, work_dir=tmp_path / "out",
        compile_tool=str(compile_tool),
    )
    compile_result = ModelPipeline(config).compile()
    assert compile_result.compile_trace is None
    assert compile_result.perfetto_viewer is None


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
