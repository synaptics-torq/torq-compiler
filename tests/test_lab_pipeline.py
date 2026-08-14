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

import numpy as np
import pytest

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
