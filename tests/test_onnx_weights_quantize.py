# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Unit tests for the weight-only ONNX quantization module (ModelProto and file I/O)."""

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from torq.lab.quantization.onnx.weights import (
    onnx_weights_quantize,
    onnx_weights_quantize_file,
)


def _make_matmul_model() -> onnx.ModelProto:
    weight = numpy_helper.from_array(
        np.random.randn(64, 16).astype(np.float32), "weight"
    )
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weight"], ["output"], name="MatMul_0")],
        "weights-quant-matmul",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [2, 64])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [2, 16])],
        [weight],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 10
    return model


def _dql_nodes(model: onnx.ModelProto):
    return [n for n in model.graph.node if n.op_type == "DequantizeLinear"]


def test_onnx_weights_quantize_in_memory():
    """onnx_weights_quantize takes and returns a ModelProto (no files)."""
    quantized = onnx_weights_quantize(_make_matmul_model(), bits=8)
    assert isinstance(quantized, onnx.ModelProto)
    assert len(_dql_nodes(quantized)) == 1
    inits = {i.name: i for i in quantized.graph.initializer}
    assert inits["weight_quantized"].data_type == TensorProto.INT8
    assert inits["weight_scales"].data_type == TensorProto.BFLOAT16
    assert inits["weight_zero_points"].data_type == TensorProto.INT8


def test_onnx_weights_quantize_int4_in_memory():
    quantized = onnx_weights_quantize(_make_matmul_model(), bits=4)
    inits = {i.name: i for i in quantized.graph.initializer}
    assert inits["weight_quantized"].data_type == TensorProto.INT4
    assert inits["weight_scales"].data_type == TensorProto.BFLOAT16
    assert inits["weight_zero_points"].data_type == TensorProto.INT4


def test_onnx_weights_quantize_dequantize_bf16():
    """dequantize_weights bakes the error into bf16 weights (no DQL nodes)."""
    quantized = onnx_weights_quantize(
        _make_matmul_model(), bits=8, dequantize_weights=True
    )
    assert not _dql_nodes(quantized)
    assert (
        quantized.graph.input[0].type.tensor_type.elem_type == TensorProto.BFLOAT16
    )
    assert (
        quantized.graph.output[0].type.tensor_type.elem_type == TensorProto.BFLOAT16
    )
    assert any(
        i.data_type == TensorProto.BFLOAT16 for i in quantized.graph.initializer
    )


def test_onnx_weights_quantize_file(tmp_path):
    """onnx_weights_quantize_file takes and writes file paths."""
    model_path = tmp_path / "model.onnx"
    output_path = tmp_path / "model_weights.onnx"
    onnx.save(_make_matmul_model(), model_path)

    returned = onnx_weights_quantize_file(model_path, output_path, bits=8)

    assert returned == output_path
    assert output_path.exists()
    assert _dql_nodes(onnx.load(str(output_path)))


def test_onnx_weights_quantize_matches_file_flow(tmp_path):
    """Both I/O variants run the same core pass and agree on the graph."""
    in_memory = onnx_weights_quantize(_make_matmul_model(), bits=4)
    model_path = tmp_path / "model.onnx"
    output_path = tmp_path / "model_weights.onnx"
    onnx.save(_make_matmul_model(), model_path)
    onnx_weights_quantize_file(model_path, output_path, bits=4)
    from_file = onnx.load(str(output_path))

    assert [n.op_type for n in in_memory.graph.node] == [
        n.op_type for n in from_file.graph.node
    ]
