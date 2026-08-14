# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Tests for the torq.lab.reference numpy/ONNX reference impls.

Skips cleanly when the onnx extra is not installed (torq.lab.reference imports
onnx at module scope), so the base-wheel torq-lab CI lane does not error.
"""

import numpy as np
import pytest

pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from torq.lab import reference  # noqa: E402


def test_torch_tanh_gelu_numpy_matches_reference_points():
    x = np.array([-1.0, 0.0, 1.0, 3.0], np.float32)
    y = reference.torch_tanh_gelu_numpy(x)
    assert abs(y[1]) < 1e-6            # gelu(0) == 0
    assert y[2] == pytest.approx(0.8411919, abs=1e-4)
    assert y[3] == pytest.approx(2.9963627, abs=1e-3)


def test_torch_tanh_gelu_numpy_output_dtype():
    x = np.array([1.0, 2.0], np.float32)
    y = reference.torch_tanh_gelu_numpy(x, output_dtype=np.float16)
    assert y.dtype == np.float16


def test_numpy_maxpool_2x2():
    x = np.arange(16, dtype=np.float32).reshape(1, 1, 4, 4)
    out = reference._numpy_maxpool(x, (2, 2), (2, 2), (0, 0, 0, 0))
    assert out.shape == (1, 1, 2, 2)
    # max of each 2x2 block of a row-major 0..15 grid
    assert out[0, 0].tolist() == [[5.0, 7.0], [13.0, 15.0]]


def test_numpy_global_average_pool():
    x = np.array([[[[1.0, 3.0], [5.0, 7.0]]]], np.float32)  # (1,1,2,2), mean=4
    out = reference._numpy_global_average_pool(x)
    assert out.shape == (1, 1, 1, 1)
    assert out[0, 0, 0, 0] == pytest.approx(4.0)


def test_numpy_instance_norm_zero_mean_unit_var():
    x = np.array([[[1.0, 2.0, 3.0, 4.0]]], np.float32)  # (1,1,4)
    scale = np.array([1.0], np.float32)
    bias = np.array([0.0], np.float32)
    out = reference._numpy_instance_norm(x, scale, bias, 0.0, np.float32)
    assert out.mean() == pytest.approx(0.0, abs=1e-5)
    assert out.std() == pytest.approx(1.0, abs=1e-3)


def test_has_gelu_detects_node():
    from onnx import helper, TensorProto

    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])
    node = helper.make_node("Gelu", ["x"], ["y"], approximate="tanh")
    model = helper.make_model(helper.make_graph([node], "g", [x], [y]))
    assert reference._has_gelu(model) is True

    node2 = helper.make_node("Relu", ["x"], ["y"])
    model2 = helper.make_model(helper.make_graph([node2], "g", [x], [y]))
    assert reference._has_gelu(model2) is False
