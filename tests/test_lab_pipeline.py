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

import ml_dtypes
import numpy as np
import pytest

from _lab_fake_tools import write_fake_compile, write_fake_run
from torq.lab.pipeline import ModelPipeline
from torq.lab.types import PipelineConfig, RunResult


def test_random_inputs_without_spec_raises(tmp_path):
    vmfb = tmp_path / "model.vmfb"
    vmfb.write_bytes(b"FAKEVMFB")
    config = PipelineConfig(model_path=vmfb, work_dir=tmp_path / "out",
                            run_tool="torq-run-module", random_inputs=True)
    with pytest.raises(Exception):
        ModelPipeline(config).run(vmfb)


def test_compare_none_when_no_expected(tmp_path):
    config = PipelineConfig(model_path=tmp_path / "m.mlir", work_dir=tmp_path / "out")
    pipe = ModelPipeline(config)
    assert pipe.compare(RunResult(command=[], outputs=[np.zeros(4, np.float32)])) is None


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
