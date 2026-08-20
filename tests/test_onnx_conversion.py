"""Unit tests for the ONNX-level dtype conversions.

Covers torq.testing.onnx.convert_fp32_to_bf16 and
torq.testing.onnx.convert_int64_to_int32 with small synthetic models.

Run: pytest tests/test_onnx_conversion.py -v
"""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from torq.testing.onnx import (
    convert_fp32_to_bf16,
    convert_int64_to_int32,
    is_model_int32,
)


# ---------------------------------------------------------------------------
# FP32 -> BF16
# ---------------------------------------------------------------------------

def _make_bf16_constant_model():
    """Two Constant nodes as graph outputs: a multi-element table + a scalar."""
    table_vals = np.linspace(-2.0, 2.0, 32, dtype=np.float32).reshape(4, 8)
    table = helper.make_node(
        "Constant", [], ["table_out"],
        value=numpy_helper.from_array(table_vals, "table_out"),
    )
    epsilon_vals = np.array([1e-12], dtype=np.float32)
    epsilon = helper.make_node(
        "Constant", [], ["epsilon_out"],
        value=numpy_helper.from_array(epsilon_vals, "epsilon_out"),
    )

    table_out = helper.make_tensor_value_info("table_out", TensorProto.FLOAT, [4, 8])
    epsilon_out = helper.make_tensor_value_info("epsilon_out", TensorProto.FLOAT, [1])

    graph = helper.make_graph(
        [table, epsilon], "bf16_constant_model", [], [table_out, epsilon_out]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)])
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def _bf16_constant_payloads(model):
    """Map Constant node output name -> (data_type, value ndarray)."""
    payloads = {}
    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attr in node.attribute:
            if attr.name == "value" and attr.HasField("t"):
                payloads[node.output[0]] = (
                    attr.t.data_type,
                    numpy_helper.to_array(attr.t),
                )
    return payloads


def test_bf16_constant_node_multi_element_converted():
    """Multi-element f32 Constant payloads are narrowed to bf16."""
    converted = convert_fp32_to_bf16(_make_bf16_constant_model())
    dtype, values = _bf16_constant_payloads(converted)["table_out"]

    assert dtype == TensorProto.BFLOAT16

    # numpy_helper.to_array promotes the bf16 payload back to a bfloat16 dtype;
    # widen to float32 for the round-trip check against the original.
    # Truncation drops the low 16 mantissa bits, so max rel error is 2^-7.
    original = np.linspace(-2.0, 2.0, 32, dtype=np.float32).reshape(4, 8)
    assert values.astype(np.float32).shape == original.shape
    assert np.allclose(values.astype(np.float32), original, rtol=2**-7, atol=1e-3)


def test_bf16_constant_node_scalar_stays_f32():
    """Scalar f32 Constant payloads (epsilon) are deliberately left f32."""
    converted = convert_fp32_to_bf16(_make_bf16_constant_model())
    dtype, values = _bf16_constant_payloads(converted)["epsilon_out"]

    assert dtype == TensorProto.FLOAT
    assert np.array_equal(values, np.array([1e-12], dtype=np.float32))


# ---------------------------------------------------------------------------
# INT64 -> INT32
# ---------------------------------------------------------------------------

def _make_int32_gather_reshape_model(table_values=None) -> onnx.ModelProto:
    """Build a small INT64 model: Gather -> Reshape -> Cast(INT64) -> Cast(FLOAT).

    - Gather: i64 constant table + i64 indices graph input (schema accepts i32).
    - Reshape: i64 shape constant (schema requires i64 -> needs a repair cast).
    - Cast(to=INT64): retargeted to INT32 by the conversion.
    Output is float so the graph output annotation is unaffected.
    """
    if table_values is None:
        table_values = np.arange(32, dtype=np.int64).reshape(8, 4)

    table = helper.make_node(
        "Constant", [], ["table"],
        value=numpy_helper.from_array(table_values, "table"),
    )
    shape = helper.make_node(
        "Constant", [], ["shape"],
        value=numpy_helper.from_array(np.array([3, 4], dtype=np.int64), "shape"),
    )
    gather = helper.make_node("Gather", ["table", "indices"], ["gathered"], axis=0)
    reshape = helper.make_node("Reshape", ["gathered", "shape"], ["flat"])
    cast_i64 = helper.make_node("Cast", ["flat"], ["cast_i64"], to=TensorProto.INT64)
    cast_f32 = helper.make_node("Cast", ["cast_i64"], ["out"], to=TensorProto.FLOAT)

    indices = helper.make_tensor_value_info("indices", TensorProto.INT64, [1, 3])
    out = helper.make_tensor_value_info("out", TensorProto.FLOAT, [3, 4])

    graph = helper.make_graph(
        [table, shape, gather, reshape, cast_i64, cast_f32],
        "gather_reshape_i64",
        [indices],
        [out],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 18)]
    )
    model.ir_version = 8
    onnx.checker.check_model(model)
    return model


def _int32_constant_value_types(model: onnx.ModelProto) -> dict:
    """Map Constant node output name -> value tensor data type."""
    types = {}
    for node in model.graph.node:
        if node.op_type == "Constant":
            for attr in node.attribute:
                if attr.name == "value" and attr.HasField("t"):
                    types[node.output[0]] = attr.t.data_type
    return types


def test_int32_conversion_types_and_repairs(capsys):
    """Constants/inputs become int32; Reshape shape gets a Cast-back repair."""
    converted = convert_int64_to_int32(_make_int32_gather_reshape_model())

    # Constant payloads narrowed to int32.
    const_types = _int32_constant_value_types(converted)
    assert const_types == {"table": TensorProto.INT32, "shape": TensorProto.INT32}

    # Graph input annotation flipped to int32.
    assert converted.graph.input[0].type.tensor_type.elem_type == TensorProto.INT32

    # Cast(to=INT64) retargeted to INT32.
    casts = [n for n in converted.graph.node if n.op_type == "Cast"]
    original_cast = next(n for n in casts if n.output[0] == "cast_i64")
    assert original_cast.attribute[0].i == TensorProto.INT32

    # Repair: a Cast back to INT64 feeding the Reshape shape input.
    repair_casts = [n for n in casts if n.attribute[0].i == TensorProto.INT64]
    assert len(repair_casts) == 1
    assert repair_casts[0].input[0] == "shape"
    reshape = next(n for n in converted.graph.node if n.op_type == "Reshape")
    assert reshape.input[1] == repair_casts[0].output[0]

    # Nothing int64 left in initializers/inputs/outputs/Constant payloads.
    assert is_model_int32(converted)

    # Summary print mirrors the BF16 "max error" style.
    summary = capsys.readouterr().out
    assert "[INT32] Converted 2 tensors, max |value|: 31" in summary
    assert "inserted 1 Cast-to-int64 repair node(s)" in summary


def test_int32_conversion_range_gate():
    """Out-of-range constants fail hard, naming the offending tensor."""
    table = np.arange(32, dtype=np.int64).reshape(8, 4)
    table[0, 0] = 2**31  # just outside the int32 range
    model = _make_int32_gather_reshape_model(table_values=table)

    with pytest.raises(ValueError, match="Constant:table"):
        convert_int64_to_int32(model)

    table[0, 0] = -(2**31) - 1
    model = _make_int32_gather_reshape_model(table_values=table)
    with pytest.raises(ValueError, match="Constant:table"):
        convert_int64_to_int32(model)


def test_int32_conversion_bit_identical_onnxruntime(tmp_path):
    """Original vs converted model produce bit-identical outputs in ORT."""
    onnxruntime = pytest.importorskip("onnxruntime")

    original = _make_int32_gather_reshape_model()
    converted = convert_int64_to_int32(original)

    orig_path = tmp_path / "original.onnx"
    conv_path = tmp_path / "converted.onnx"
    onnx.save(original, str(orig_path))
    onnx.save(converted, str(conv_path))
    onnx.checker.check_model(converted)

    indices = np.array([[0, 5, 7]], dtype=np.int64)
    sess_orig = onnxruntime.InferenceSession(
        str(orig_path), providers=["CPUExecutionProvider"]
    )
    sess_conv = onnxruntime.InferenceSession(
        str(conv_path), providers=["CPUExecutionProvider"]
    )

    out_orig = sess_orig.run(None, {"indices": indices})
    out_conv = sess_conv.run(None, {"indices": indices.astype(np.int32)})

    assert len(out_orig) == len(out_conv) == 1
    assert out_orig[0].dtype == out_conv[0].dtype == np.float32
    assert out_orig[0].shape == out_conv[0].shape == (3, 4)
    assert out_orig[0].tobytes() == out_conv[0].tobytes(), (
        "converted model output is not bit-identical"
    )
