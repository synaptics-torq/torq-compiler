# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unit tests for torq.lab.manifest."""

import json

from torq.lab.artifact import ArtifactInfo
from torq.lab.compare import ComparisonResult
from torq.lab.manifest import (
    atomic_write_json_file,
    atomic_write_json_manifest,
    build_manifest,
)
from torq.lab.types import CompileResult, PipelineConfig, RemoteTarget, RunResult


def _config(tmp_path):
    return PipelineConfig(model_path=tmp_path / "model.mlir", work_dir=tmp_path / "out")


def test_build_manifest_full(tmp_path):
    config = _config(tmp_path)
    compile_result = CompileResult(
        vmfb_path=tmp_path / "out" / "model.vmfb",
        command=["torq-compile", "model.mlir", "-o", "model.vmfb"],
        elapsed=1.5,
        debug_dir=tmp_path / "out" / "debug",
    )
    run_result = RunResult(
        command=["torq-run-module", "--module=model.vmfb"],
        output_paths=[tmp_path / "out" / "outputs" / "output_0.bin"],
        wall_time=2.5,
        host_profile=tmp_path / "out" / "profiles" / "host_profile.csv",
    )
    remote = RemoteTarget(address="user@host", port=2222, remote_runner_path="/opt/torq-run-module")
    comparison = ComparisonResult(passed=True)

    manifest = build_manifest(
        config=config, compile_result=compile_result, run_result=run_result,
        remote=remote, comparison=comparison,
    )

    assert manifest["schema_version"] == 3
    assert manifest["status"] == "ok"
    # config and results are kept separate
    assert manifest["config"]["model_path"] == str(tmp_path / "model.mlir")
    assert manifest["config"]["remote"]["address"] == "user@host"
    assert manifest["config"]["remote"]["port"] == 2222
    assert manifest["results"]["compile"]["tool"] == "torq-compile"
    assert manifest["results"]["compile"]["elapsed_s"] == 1.5
    assert manifest["results"]["run"]["wall_time_s"] == 2.5
    assert manifest["results"]["run"]["tool"] == "torq-run-module"
    assert manifest["results"]["comparison"]["passed"] is True


def test_build_manifest_minimal(tmp_path):
    manifest = build_manifest(config=_config(tmp_path), status="error", diagnostics="boom")
    assert "compile" not in manifest["results"]
    assert "run" not in manifest["results"]
    assert "remote" not in manifest["config"]
    assert manifest["status"] == "error"
    assert manifest["results"]["diagnostics"] == "boom"


def test_config_roundtrips_through_manifest(tmp_path):
    config = PipelineConfig(
        model_path=tmp_path / "model.mlir",
        work_dir=tmp_path / "out",
        chip="SL1620",
        runtime_hw_type="aws_fpga",
        compiler_options=["--torq-foo"],
        expected_output_npy=[tmp_path / "exp0.npy"],
        reference="onnx:/abs/ref.onnx",
    )
    remote = RemoteTarget(address="user@host", port=2222, remote_runner_path="/opt/torq-run-module")
    manifest = build_manifest(config=config, remote=remote)

    assert PipelineConfig.from_manifest(manifest) == config
    assert RemoteTarget.from_manifest(manifest) == remote
    assert RemoteTarget.from_manifest(build_manifest(config=config)) is None


def test_build_manifest_with_artifact_section(tmp_path):
    info = ArtifactInfo(
        source_path=tmp_path / "m.mlir",
        source_kind="mlir",
        function="qkv",
        entry_points=["qkv"],
        converted_io=["all"],
        debug_dir=tmp_path / "out" / "debug",
        chip="SL2610",
        compile_command=["torq-compile"],
    )
    manifest = build_manifest(config=_config(tmp_path), artifact=info)
    assert manifest["artifact"]["function"] == "qkv"
    assert manifest["artifact"]["chip"] == "SL2610"
    assert manifest["artifact"]["source_kind"] == "mlir"
    assert manifest["artifact"]["compile_command"] == ["torq-compile"]


def test_atomic_write_json_roundtrip(tmp_path):
    key_dir = tmp_path / "cache" / "key"
    atomic_write_json_file(key_dir, "cases.json", {"a": 1, "b": [2, 3]})
    atomic_write_json_manifest(key_dir, {"version": 1})

    assert json.loads((key_dir / "cases.json").read_text()) == {"a": 1, "b": [2, 3]}
    assert json.loads((key_dir / "manifest.json").read_text()) == {"version": 1}
    # no temp files are left behind after the rename
    assert not list(key_dir.glob("*.tmp.*"))
