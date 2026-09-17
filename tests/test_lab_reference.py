# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the torq.lab.verification.reference numpy/ONNX reference impls.

Skips cleanly when the onnx extra is not installed (torq.lab.verification.reference imports
onnx at module scope), so the base-wheel torq-lab CI lane does not error.
"""

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from torq.lab.verification import reference  # noqa: E402


def test_torch_tanh_gelu_numpy_matches_reference_points():
    x = np.array([-1.0, 0.0, 1.0, 3.0], np.float32)
    y = reference.torch_tanh_gelu_numpy(x)
    assert abs(y[1]) < 1e-6            # gelu(0) == 0
    assert y[2] == pytest.approx(0.8411919, abs=1e-4)
    assert y[3] == pytest.approx(2.9963627, abs=1e-3)


def test_numpy_maxpool_2x2():
    x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    out = reference.numpy_maxpool(x, (2, 2), (2, 2), (0, 0, 0, 0))
    assert out.shape == (1, 1, 2, 2)
    # max of each 2x2 block of a row-major 0..15 grid
    assert out[0, 0].tolist() == [[5.0, 7.0], [13.0, 15.0]]


def test_numpy_global_average_pool():
    x = np.array([[[[1.0, 3.0], [5.0, 7.0]]]], np.float32)  # (1,1,2,2), mean=4
    out = reference.numpy_global_average_pool(x)
    assert out.shape == (1, 1, 1, 1)
    assert out[0, 0, 0, 0] == pytest.approx(4.0)


def test_numpy_instance_norm_zero_mean_unit_var():
    x = np.array([[[1.0, 2.0, 3.0, 4.0]]], np.float32)  # (1,1,4)
    scale = np.array([1.0], np.float32)
    bias = np.array([0.0], np.float32)
    out = reference.numpy_instance_norm(x, scale, bias, 0.0, np.float32)
    assert out.mean() == pytest.approx(0.0, abs=1e-5)
    assert out.std() == pytest.approx(1.0, abs=1e-3)


# A static-shape graph the llvm-cpu golden runner can compile and execute.
ADD_MLIR = """
module {
  func.func @main(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = "tosa.add"(%arg0, %arg1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}
"""


def test_llvmcpu_reference_outputs(tmp_path):
    """The llvm-cpu golden runner compiles an MLIR and returns its outputs.

    Requires iree-compile / iree-run-module (the same binaries the pytest
    flow's llvmcpu_reference_results fixture uses); skips cleanly when they are
    not installed, like the tool-dependent tests of the pipeline.
    """
    from torq.lab.pipeline import tools

    try:
        tools.find_iree_compile_tool()
        tools.find_iree_run_tool()
    except FileNotFoundError as exc:
        pytest.skip(f"iree tools not available: {exc}")

    mlir = tmp_path / "add.mlir"
    mlir.write_text(ADD_MLIR)
    a = np.arange(4, dtype=np.float32).reshape(2, 2)
    b = np.ones((2, 2), np.float32)

    outputs = reference.llvmcpu_reference_outputs(mlir, [a, b], tmp_path / "work")

    assert len(outputs) == 1
    assert outputs[0].dtype == np.float32
    assert np.array_equal(outputs[0], a + b)


def test_onnx_reference_outputs_raises_labError_naming_the_op(tmp_path, monkeypatch):
    """When both ORT and the numpy hybrid fail, the error names the op, points
    at --golden, and surfaces both tiers' underlying reasons (the onnxruntime
    error is the actionable one for an fp32 model ORT loaded but failed to run)."""
    from torq.lab import LabError

    path = tmp_path / "model.onnx"
    path.write_bytes(b"not-a-real-onnx-file")

    class FailingSession:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("Implementation for InstanceNormalization not supported")

    monkeypatch.setattr(reference.onnxruntime, "InferenceSession", FailingSession)
    monkeypatch.setattr(reference.onnx, "load", lambda p: object())

    def fake_execute(model, inputs):
        raise RuntimeError(
            "Could not execute InstanceNormalization with onnxruntime: "
            "Implementation for InstanceNormalization not supported"
        )

    monkeypatch.setattr(reference, "execute_onnx_model_numpy", fake_execute)

    with pytest.raises(LabError) as excinfo:
        reference.onnx_reference_outputs(path, [np.zeros(4, np.float32)])

    message = str(excinfo.value)
    assert "InstanceNormalization" in message
    assert "--golden" in message
    # The underlying tier errors are surfaced, not hidden: the onnxruntime
    # message is what a user needs when ORT loaded the model but failed to run.
    assert "onnxruntime:" in message
    assert "Implementation for" in message
