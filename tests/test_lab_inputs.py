# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unit tests for seeded/ranged random inputs (PipelineConfig -> lab.io)."""

import json

import ml_dtypes
import numpy as np
import pytest

from torq.lab import io
from torq.lab.pipeline import ModelPipeline
from torq.lab.types import LabError, MlirIoSpec, PipelineConfig, TensorType


SPEC = MlirIoSpec(
    inputs=[
        TensorType([2, 3], "f32"),
        TensorType([4, 4], "i32"),
        TensorType([8], "ui8"),
        TensorType([5], "i1"),
    ],
    outputs=[],
)


def _config(tmp_path, **overrides):
    return PipelineConfig(
        model_path=tmp_path / "model.mlir",
        work_dir=tmp_path / "work",
        random_inputs=True,
        **overrides,
    )


# -- seeding -----------------------------------------------------------------


def test_fixed_seed_is_deterministic():
    a = io.generate_random_inputs(SPEC, seed=42)
    b = io.generate_random_inputs(SPEC, seed=42)
    assert len(a) == len(b) == len(SPEC.inputs)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


# -- per-input ranges ----------------------------------------------------------


def test_range_clamping_per_input():
    ranges = {"0": (-1.0, 1.0), "1": (-5, 5)}
    inputs = io.generate_random_inputs(SPEC, seed=7, ranges=ranges)
    f32_data, i32_data, ui8_data, _ = inputs
    assert f32_data.min() >= -1.0 and f32_data.max() < 1.0
    assert i32_data.min() >= -5 and i32_data.max() < 5  # max exclusive
    # Inputs without a matching key keep the full dtype range.
    assert ui8_data.dtype == np.uint8


def test_range_bf16_input():
    spec = MlirIoSpec(inputs=[TensorType([16], "bf16")], outputs=[])
    (data,) = io.generate_random_inputs(spec, seed=3, ranges={"0": (-2.0, 2.0)})
    assert data.dtype == np.dtype(ml_dtypes.bfloat16)
    as_f32 = data.astype(np.float32)
    assert as_f32.min() >= -2.0 and as_f32.max() < 2.0


def test_bool_inputs_ignore_range():
    # Mirrors tweaked_random_input_data: bools always draw from {0, 1}.
    inputs = io.generate_random_inputs(SPEC, seed=5, ranges={"3": (0, 100)})
    bool_data = inputs[3]
    assert bool_data.dtype == np.dtype(bool)
    assert set(np.unique(bool_data)) <= {False, True}


def test_unknown_range_key_raises():
    with pytest.raises(LabError, match="match no model input"):
        io.generate_random_inputs(SPEC, ranges={"9": (0, 1)})


def test_malformed_range_raises():
    with pytest.raises(LabError, match="min < max"):
        io.generate_random_inputs(SPEC, ranges={"0": (1.0, 1.0)})


# -- PipelineConfig wiring -----------------------------------------------------


def test_pipeline_config_json_roundtrip(tmp_path):
    config = _config(
        tmp_path, input_seed=99, input_ranges={"0": (0.0, 1.0), "1": (0, 640)}
    )
    restored = PipelineConfig.from_dict(json.loads(json.dumps(config.to_dict())))
    assert restored.input_seed == 99
    assert restored.input_ranges == {"0": (0.0, 1.0), "1": (0, 640)}
    # Round-tripped config generates the same data.
    for x, y in zip(
        io.generate_random_inputs(SPEC, seed=config.input_seed, ranges=config.input_ranges),
        io.generate_random_inputs(SPEC, seed=restored.input_seed, ranges=restored.input_ranges),
    ):
        assert np.array_equal(x, y)


def test_materialize_inputs_honors_seed_and_ranges(tmp_path):
    spec = MlirIoSpec(inputs=[TensorType([4, 4], "i32"), TensorType([2], "f32")], outputs=[])
    pipeline = ModelPipeline(_config(tmp_path, input_seed=11, input_ranges={"0": (0, 3)}))
    inputs, paths = pipeline._materialize_inputs(spec, pipeline._convert_io_dtypes_policy())
    assert inputs[0].min() >= 0 and inputs[0].max() < 3
    assert [p.name for p in paths] == ["in_rnd_0.bin", "in_rnd_1.bin"]

    # Same seed reproduces; a different seed differs.
    p_again = ModelPipeline(_config(tmp_path, input_seed=11, input_ranges={"0": (0, 3)}))
    again, _ = p_again._materialize_inputs(spec, p_again._convert_io_dtypes_policy())
    p_other = ModelPipeline(_config(tmp_path, input_seed=12, input_ranges={"0": (0, 3)}))
    other, _ = p_other._materialize_inputs(spec, p_other._convert_io_dtypes_policy())
    for x, y in zip(inputs, again):
        assert np.array_equal(x, y)
    assert any(not np.array_equal(x, y) for x, y in zip(inputs, other))


