# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unit tests for the dynamic ONNX quantization module (ModelProto and file I/O)."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from torq.lab.quantization.onnx.dynamic import (
    onnx_dynamic_quantize,
    onnx_dynamic_quantize_file,
)
from torq.lab.quantization.onnx.static import is_model_quantized


def _make_matmul_model() -> onnx.ModelProto:
    weight = numpy_helper.from_array(np.eye(2, dtype=np.float32), "weight")
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weight"], ["output"], name="MatMul_0")],
        "dynamic-quant-matmul",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    return model


def test_onnx_dynamic_quantize_in_memory():
    """onnx_dynamic_quantize takes and returns a ModelProto (no files)."""
    quantized = onnx_dynamic_quantize(_make_matmul_model())
    assert isinstance(quantized, onnx.ModelProto)
    assert is_model_quantized(quantized)


def test_onnx_dynamic_quantize_file(tmp_path):
    """onnx_dynamic_quantize_file takes and writes file paths."""
    model_path = tmp_path / "model.onnx"
    output_path = tmp_path / "model_dynamic.onnx"
    onnx.save(_make_matmul_model(), model_path)

    returned = onnx_dynamic_quantize_file(model_path, output_path)

    assert returned == output_path
    assert output_path.exists()
    assert is_model_quantized(onnx.load(str(output_path)))


def test_onnx_dynamic_quantize_matches_file_flow(tmp_path):
    """Both I/O variants run the same core pass and agree on the graph."""
    in_memory = onnx_dynamic_quantize(_make_matmul_model())
    model_path = tmp_path / "model.onnx"
    output_path = tmp_path / "model_dynamic.onnx"
    onnx.save(_make_matmul_model(), model_path)
    onnx_dynamic_quantize_file(model_path, output_path)
    from_file = onnx.load(str(output_path))

    assert [n.op_type for n in in_memory.graph.node] == [
        n.op_type for n in from_file.graph.node
    ]
