# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unit tests for torq.lab.io and torq.lab.types."""

import ml_dtypes
import numpy as np
import pytest

from torq.lab import io
from torq.lab.types import LabError, MlirIoSpec, TensorType


@pytest.mark.parametrize(
    "name,dtype",
    [
        ("f32", np.float32),
        ("bf16", ml_dtypes.bfloat16),
        ("si8", np.int8),
        ("i32", np.int32),
    ],
)
def test_get_dtype(name, dtype):
    assert io.get_dtype(name) == dtype


def test_get_dtype_unsupported():
    with pytest.raises(LabError):
        io.get_dtype("f64")


@pytest.mark.parametrize("fmt,dtype", [("f32", np.float32), ("bf16", ml_dtypes.bfloat16)])
def test_load_outputs_roundtrip(tmp_path, fmt, dtype):
    spec = TensorType([2, 3], fmt)
    ref = (np.arange(6).reshape(2, 3)).astype(dtype)
    out_path = tmp_path / "output_0.bin"
    out_path.write_bytes(ref.tobytes())

    loaded = io.load_outputs([spec], [out_path])
    assert len(loaded) == 1
    assert loaded[0].shape == (2, 3)
    assert loaded[0].dtype == np.dtype(dtype)
    assert np.array_equal(loaded[0], ref)


def test_write_inputs_and_build_args_with_spec(tmp_path):
    spec = MlirIoSpec(inputs=[TensorType([2, 2], "i8")], outputs=[])
    inputs = io.generate_random_inputs(spec)
    paths = io.write_inputs(inputs, tmp_path / "inputs")
    assert paths[0].name == "in_rnd_0.bin"
    assert (tmp_path / "inputs" / "in_rnd_0.bin.npy").exists()

    args = io.build_input_args(paths, inputs, spec)
    assert args == [f"--input=2x2xi8=@{paths[0]}"]


TOSA_MLIR = """
module {
  func.func @main(%arg0: tensor<1x4xf32>, %arg1: tensor<2xi32>) -> tensor<1x4xf32> {
    return %arg0 : tensor<1x4xf32>
  }
}
"""

TORCH_MLIR = """
module {
  func.func @forward(%arg0: !torch.vtensor<[2,3],si32>) -> !torch.vtensor<[2,3],si32> {
    return %arg0 : !torch.vtensor<[2,3],si32>
  }
}
"""

# Modules with more than one top-level function (jax lowerings, for instance,
# emit several) must resolve to a single entry function's IO, not the union of
# every function's arguments/results. The last function op wins, matching the
# entry point the runner executes.
MULTI_FUNC_MLIR = """
module {
  func.func @helper(%arg0: tensor<7xf32>) -> tensor<7xf32> {
    return %arg0 : tensor<7xf32>
  }
  func.func @main(%arg0: tensor<1xf32>) -> tensor<1xf32> {
    return %arg0 : tensor<1xf32>
  }
}
"""


def test_parse_mlir_io_spec_tosa(tmp_path):
    path = tmp_path / "tosa.mlir"
    path.write_text(TOSA_MLIR)
    spec = io.parse_mlir_io_spec(path)
    assert spec.inputs == [TensorType([1, 4], "f32"), TensorType([2], "i32")]
    assert spec.outputs == [TensorType([1, 4], "f32")]


def test_parse_mlir_io_spec_multi_function_single_entry(tmp_path):
    # Regression: the spec must describe one entry function, not accumulate the
    # arguments/results of every top-level function. A jax `x + 1` lowering
    # emits multiple functions; accumulating produced two 1xf32 inputs/outputs
    # and handed iree-run-module the wrong argument count.
    path = tmp_path / "multi.mlir"
    path.write_text(MULTI_FUNC_MLIR)
    spec = io.parse_mlir_io_spec(path)
    assert spec.inputs == [TensorType([1], "f32")]
    assert spec.outputs == [TensorType([1], "f32")]


def test_parse_mlir_io_spec_torch(tmp_path):
    path = tmp_path / "torch.mlir"
    path.write_text(TORCH_MLIR)
    try:
        spec = io.parse_mlir_io_spec(path)
    except Exception as exc:  # torch dialect may not be registered in this context
        pytest.skip(f"torch dialect not parseable in this environment: {exc}")
    assert spec.inputs == [TensorType([2, 3], "si32")]
    assert spec.outputs == [TensorType([2, 3], "si32")]
