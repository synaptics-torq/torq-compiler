# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unit tests for ModelPipeline.compare() and run-input guards.

Command assembly and local compile/run plumbing are exercised by real usage
(every model compile/run goes through ModelPipeline) and by the profiling
wiring tests; this file keeps only the behaviours a green end-to-end run would
not catch: the compare gate reporting a failure, and the missing-spec guard.
"""

import json
from pathlib import Path

import ml_dtypes
import numpy as np
import pytest

from _lab_fake_tools import write_fake_compile, write_fake_run
from torq.lab import LabError
from torq.lab.pipeline.workflow import ModelPipeline, PipelineConfig, RunResult

TOSA_MLIR = """
module {
  func.func @main(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
    return %arg0 : tensor<1x4xf32>
  }
}
"""


def test_random_inputs_without_spec_raises(tmp_path):
    vmfb = tmp_path / "model.vmfb"
    vmfb.write_bytes(b"FAKEVMFB")
    config = PipelineConfig(model_path=vmfb, work_dir=tmp_path / "out",
                            run_tool="torq-run-module", random_inputs=True)
    with pytest.raises(Exception):
        ModelPipeline(config).run(vmfb)


def test_compare_flags_missing_outputs(tmp_path):
    # A run that produced no loadable outputs while expected outputs were
    # configured must be reported as a failure, not silently pass.
    exp = tmp_path / "exp0.npy"
    np.save(exp, np.array([1.0, 2.0, 3.0, 4.0], np.float32))
    config = PipelineConfig(model_path=tmp_path / "m.mlir", work_dir=tmp_path / "out",
                            expected_output_npy=[exp])
    result = ModelPipeline(config).compare(RunResult(command=[], outputs=[]))
    assert result is not None and not result.passed
    assert "number of outputs differ" in result.reason


def test_compile_run_converts_public_io_dtypes(tmp_path, monkeypatch):
    compile_tool = write_fake_compile(tmp_path / "tools")
    run_tool = write_fake_run(tmp_path / "tools")
    run_record = tmp_path / "run_argv.txt"
    monkeypatch.setenv("TORQ_COMPILE", str(compile_tool))
    monkeypatch.setenv("TORQ_RUN_MODULE", str(run_tool))
    monkeypatch.setenv("FAKE_RUN_RECORD", str(run_record))
    monkeypatch.setenv("FAKE_OUTPUT_SIZES", "8")  # 1x4xbf16

    model = tmp_path / "model.mlir"
    model.write_text("""
module {
  func.func @main(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
    return %arg0 : tensor<1x4xf32>
  }
}
""")
    input_path = tmp_path / "input.npy"
    np.save(input_path, np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
    work_dir = tmp_path / "out"
    config = PipelineConfig(
        model_path=model,
        work_dir=work_dir,
        compiler_options=["--torq-convert-dtypes", "--torq-convert-io-dtype"],
        input_npy=[input_path],
    )

    _, run_result = ModelPipeline(config).compile_run()

    assert f"--input=1x4xbf16=@{work_dir / 'inputs' / 'in_rnd_0.bin'}" in run_record.read_text()
    materialized_input = np.fromfile(
        work_dir / "inputs" / "in_rnd_0.bin", dtype=ml_dtypes.bfloat16
    )
    assert np.array_equal(
        materialized_input, np.array([1.0, 2.0, 3.0, 4.0], dtype=ml_dtypes.bfloat16)
    )
    assert run_result.outputs[0].dtype == np.dtype(ml_dtypes.bfloat16)


def test_run_precompiled_vmfb_converts_public_io_dtypes(tmp_path, monkeypatch):
    run_tool = write_fake_run(tmp_path / "tools")
    run_record = tmp_path / "run_argv.txt"
    monkeypatch.setenv("TORQ_RUN_MODULE", str(run_tool))
    monkeypatch.setenv("FAKE_RUN_RECORD", str(run_record))
    monkeypatch.setenv("FAKE_OUTPUT_SIZES", "8")  # 1x4xbf16

    model = tmp_path / "model.mlir"
    model.write_text("""
module {
  func.func @main(%arg0: tensor<1x4xf32>) -> tensor<1x4xf32> {
    return %arg0 : tensor<1x4xf32>
  }
}
""")
    vmfb = model.with_suffix(".vmfb")
    vmfb.write_bytes(b"FAKEVMFB")
    work_dir = tmp_path / "out"
    config = PipelineConfig(
        model_path=vmfb,
        work_dir=work_dir,
        random_inputs=True,
        convert_io_dtypes=["all"],
    )

    run_result = ModelPipeline(config).run()

    assert f"--input=1x4xbf16=@{work_dir / 'inputs' / 'in_rnd_0.bin'}" in run_record.read_text()
    assert run_result.outputs[0].dtype == np.dtype(ml_dtypes.bfloat16)


def test_vmfb_path_precedence_config_vmfb_path(tmp_path):
    # config.vmfb_path wins over model_path.
    vmfb = tmp_path / "model.vmfb"
    vmfb.write_bytes(b"FAKEVMFB")
    config_vmfb = tmp_path / "config.vmfb"
    config_vmfb.write_bytes(b"CONFIG")
    work_dir = tmp_path / "out"
    config = PipelineConfig(model_path=vmfb, work_dir=work_dir, vmfb_path=config_vmfb)
    pipe = ModelPipeline(config)
    assert pipe._vmfb_path() == config_vmfb.resolve()


def test_ensure_mlir_rejects_vmfb(tmp_path):
    vmfb = tmp_path / "model.vmfb"
    vmfb.write_bytes(b"FAKEVMFB")
    config = PipelineConfig(model_path=vmfb, work_dir=tmp_path / "w")
    pipe = ModelPipeline(config)
    with pytest.raises(LabError, match=r"\.onnx.*\.tflite.*\.mlir"):
        pipe.ensure_mlir()


def test_ensure_mlir_imports_onnx(tmp_path, monkeypatch):
    onnx_path = tmp_path / "m.onnx"
    onnx_path.write_bytes(b"fake-onnx")
    calls = {}

    def fake_convert(src, out, timeout=300):
        Path(out).write_text(TOSA_MLIR)
        calls["src"], calls["out"] = Path(src), Path(out)

    monkeypatch.setattr("torq.lab.model_tools.importers.onnx.convert_onnx_to_mlir", fake_convert)
    cfg = PipelineConfig(model_path=onnx_path, work_dir=tmp_path / "w")
    pipe = ModelPipeline(cfg)
    mlir = pipe.ensure_mlir()
    assert mlir == (tmp_path / "w" / "m.mlir")
    assert mlir.read_text() == TOSA_MLIR
    assert cfg.spec_source == mlir


def test_run_resolves_function_and_spec_from_colocated_manifest(tmp_path, monkeypatch):
    # A work-dir VMFB with only a manifest.json (no sibling .mlir) must still
    # resolve its entry function and I/O spec, via artifact.describe().
    run_tool = write_fake_run(tmp_path / "tools")
    run_record = tmp_path / "run_argv.txt"
    monkeypatch.setenv("TORQ_RUN_MODULE", str(run_tool))
    monkeypatch.setenv("FAKE_RUN_RECORD", str(run_record))
    monkeypatch.setenv("FAKE_OUTPUT_SIZES", "16")  # 1x4xf32

    work_dir = tmp_path / "art"
    work_dir.mkdir()
    vmfb = work_dir / "model.vmfb"
    vmfb.write_bytes(b"FAKEVMFB")
    (work_dir / "manifest.json").write_text(json.dumps({
        "schema_version": 3,
        "config": {},
        "results": {},
        "artifact": {
            "function": "qkv",
            "io_spec": {
                "inputs": [{"shape": [1, 4], "fmt": "f32", "name": None}],
                "outputs": [{"shape": [1, 4], "fmt": "f32", "name": None}],
            },
        },
    }))

    config = PipelineConfig(model_path=vmfb, work_dir=work_dir, random_inputs=True)
    run_result = ModelPipeline(config).run()

    assert "--function=qkv" in run_record.read_text()
    assert run_result.outputs[0].shape == (1, 4)


def test_verify_outputs_prefers_golden_over_reference(tmp_path, monkeypatch):
    exp = tmp_path / "exp0.npy"
    np.save(exp, np.zeros(4, np.float32))
    work_dir = tmp_path / "out"
    (work_dir / "inputs").mkdir(parents=True)

    def fail_reference(*args, **kwargs):
        raise AssertionError("reference generator must not run when expected_output_npy is set")

    monkeypatch.setattr("torq.lab.verification.reference.onnx_reference_outputs", fail_reference)

    config = PipelineConfig(
        model_path=tmp_path / "model.onnx", work_dir=work_dir,
        expected_output_npy=[exp], reference="onnx",
    )
    result = ModelPipeline(config).verify_outputs(RunResult(command=[], outputs=[np.zeros(4, np.float32)]))
    assert result.passed


def test_verify_outputs_falls_back_to_reference_and_reports_mismatch(tmp_path, monkeypatch):
    work_dir = tmp_path / "out"
    (work_dir / "inputs").mkdir(parents=True)
    monkeypatch.setattr("torq.lab.verification.reference.onnx_reference_outputs", lambda p, i: [np.ones(4, np.float32)])

    config = PipelineConfig(model_path=tmp_path / "model.onnx", work_dir=work_dir, reference="onnx")
    result = ModelPipeline(config).verify_outputs(RunResult(command=[], outputs=[np.zeros(4, np.float32)]))
    assert result is not None and not result.passed
