# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Regression tests for the ONNX dtype conversion command."""

import logging

import numpy as np
import onnx
import onnx_graphsurgeon as gs
import pytest
from onnx import TensorProto, helper, numpy_helper

from torq.lab.model_tools.dtype_conversion.onnx import (
    FP32Converter,
    Int64Converter,
    convert_model,
    convert_onnx_model,
)


def _convert(tmp_path, name, graph, dtype, **kwargs):
    source = tmp_path / f"{name}.onnx"
    output = tmp_path / f"{name}-{dtype}.onnx"
    onnx.save(
        helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), source
    )
    convert_model(source, output, dtype, target_opset=22, **kwargs)
    return onnx.load(output)


def _schema_node(op, input_dtypes, output_dtypes):
    return gs.Node(
        op=op,
        inputs=[gs.Variable(f"input_{idx}", dtype=dtype) for idx, dtype in enumerate(input_dtypes)],
        outputs=[gs.Variable(f"output_{idx}", dtype=dtype) for idx, dtype in enumerate(output_dtypes)],
    )


@pytest.mark.parametrize(
    ("converter", "node", "expected"),
    [
        (
            FP32Converter("bf16"),
            _schema_node("Celu", [TensorProto.FLOAT], [TensorProto.FLOAT]),
            ((0,), (0,)),
        ),
        (
            FP32Converter("bf16"),
            _schema_node("Einsum", [TensorProto.FLOAT, TensorProto.FLOAT], [TensorProto.FLOAT]),
            ((0, 1), (0,)),
        ),
        (
            FP32Converter("fp16"),
            _schema_node(
                "LayerNormalization",
                [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT],
                [TensorProto.FLOAT, TensorProto.FLOAT, TensorProto.FLOAT],
            ),
            ((), (1, 2)),
        ),
        (
            Int64Converter("int32"),
            _schema_node(
                "Col2Im",
                [TensorProto.FLOAT, TensorProto.INT64, TensorProto.INT64],
                [TensorProto.FLOAT],
            ),
            ((1, 2), ()),
        ),
        (
            Int64Converter("int16"),
            _schema_node("MatMul", [TensorProto.INT64, TensorProto.INT64], [TensorProto.INT64]),
            ((0, 1), (0,)),
        ),
    ],
)
def test_schema_enforced_io_covers_fixed_and_narrowing_contracts(converter, node, expected):
    converter._opset_version = 22
    assert converter._get_enforced_io(node) == expected


def test_preserves_cast_graph_output_dtype_without_convert_io(tmp_path):
    graph = helper.make_graph(
        [helper.make_node("Cast", ["x"], ["y"], to=TensorProto.FLOAT)],
        "public-cast",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )

    model = _convert(tmp_path, "public-cast", graph, "fp16")

    assert model.graph.output[0].type.tensor_type.elem_type == TensorProto.FLOAT
    assert any(node.op_type == "Cast" and node.output == ["y"] for node in model.graph.node)


def test_converts_in_memory_model():
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "x"], ["y"])],
        "in-memory",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    source = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])

    converted = convert_onnx_model(
        source,
        "bf16",
        convert_io=True,
        target_opset=22,
        remove_unused_node_outputs=False,
    )

    assert converted.graph.input[0].type.tensor_type.elem_type == TensorProto.BFLOAT16
    assert converted.graph.output[0].type.tensor_type.elem_type == TensorProto.BFLOAT16


def test_reports_constant_conversion_error_metrics(tmp_path, caplog):
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "c"], ["y"])],
        "constant-metrics",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
        [numpy_helper.from_array(np.array([1.234567, -2.345678], dtype=np.float32), "c")],
    )

    with caplog.at_level(logging.INFO, logger="ONNX-Dtype-Converter"):
        _convert(tmp_path, "constant-metrics", graph, "bf16", convert_io=True)

    assert "Constant 'c': values=2 max_abs_error=" in caplog.text
    assert "mean_abs_error=" in caplog.text
    assert "rmse=" in caplog.text
    assert "max_rel_error=" in caplog.text


def test_clamps_int64_sentinel_constants_for_int32(tmp_path):
    values = np.array([np.iinfo(np.int64).min, np.iinfo(np.int64).max], dtype=np.int64)
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "c"], ["y"])],
        "int64-sentinels",
        [helper.make_tensor_value_info("x", TensorProto.INT64, [2])],
        [helper.make_tensor_value_info("y", TensorProto.INT64, [2])],
        [numpy_helper.from_array(values, "c")],
    )

    model = _convert(tmp_path, "int64-sentinels", graph, "int32", convert_io=True)

    constant = next(item for item in model.graph.initializer if item.name == "c_int32")
    assert numpy_helper.to_array(constant).tolist() == [np.iinfo(np.int32).min, np.iinfo(np.int32).max]


def test_converts_constant_feeding_cast(tmp_path):
    graph = helper.make_graph(
        [
            helper.make_node("Cast", ["c"], ["cast_out"], to=TensorProto.INT64),
            helper.make_node("Add", ["x", "cast_out"], ["y"]),
        ],
        "cast-constant",
        [helper.make_tensor_value_info("x", TensorProto.INT64, [1])],
        [helper.make_tensor_value_info("y", TensorProto.INT64, [1])],
        [numpy_helper.from_array(np.array([42], dtype=np.int64), "c")],
    )

    model = _convert(tmp_path, "cast-constant", graph, "int32")

    initializer = next(item for item in model.graph.initializer if item.name == "c_int32")
    assert numpy_helper.to_array(initializer).dtype == np.int32
    assert numpy_helper.to_array(initializer).tolist() == [42]


def test_default_cleanup_preserves_unused_topk_values(tmp_path):
    graph = helper.make_graph(
        [helper.make_node("TopK", ["x", "k"], ["values", "indices"], axis=0)],
        "topk-unused-values",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("indices", TensorProto.INT64, [1])],
        [numpy_helper.from_array(np.array([1], dtype=np.int64), "k")],
    )

    model = _convert(tmp_path, "topk-unused-values", graph, "int32")

    topk = next(node for node in model.graph.node if node.op_type == "TopK")
    assert topk.output == ["values", "indices"]


@pytest.mark.parametrize(
    ("dtype", "onnx_dtype"),
    [("bf16", TensorProto.BFLOAT16), ("fp16", TensorProto.FLOAT16)],
)
def test_converts_dynamic_quantize_scale_consumers(tmp_path, dtype, onnx_dtype):
    graph = helper.make_graph(
        [
            helper.make_node("DynamicQuantizeLinear", ["x"], ["quantized", "scale", "zero_point"]),
            helper.make_node("Mul", ["scale", "weight_scale"], ["y"]),
        ],
        "dynamic-quantize-scale",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        [numpy_helper.from_array(np.array([0.25], dtype=np.float32), "weight_scale")],
    )

    model = _convert(tmp_path, f"dynamic-quantize-scale-{dtype}", graph, dtype, convert_io=True)

    onnx.checker.check_model(model, full_check=True)
    assert model.graph.output[0].type.tensor_type.elem_type == onnx_dtype
    mul = next(node for node in model.graph.node if node.op_type == "Mul")
    assert all(f"scale_{dtype}" in value for value in mul.input)


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        ("fp16", np.array([1e9, 0.5], dtype=np.float32)),
        ("int32", np.array([5_000_000_000], dtype=np.int64)),
    ],
)
def test_rejects_constants_that_overflow_export_dtype(tmp_path, dtype, values):
    original_dtype = TensorProto.FLOAT if dtype == "fp16" else TensorProto.INT64
    graph = helper.make_graph(
        [helper.make_node("Add", ["x", "c"], ["y"])],
        "constant-overflow",
        [helper.make_tensor_value_info("x", original_dtype, list(values.shape))],
        [helper.make_tensor_value_info("y", original_dtype, list(values.shape))],
        [numpy_helper.from_array(values, "c")],
    )

    with pytest.raises(ValueError, match="exceed max_float or the export dtype range"):
        _convert(tmp_path, f"constant-overflow-{dtype}", graph, dtype)
