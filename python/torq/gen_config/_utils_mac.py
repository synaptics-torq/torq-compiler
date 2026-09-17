"""Shared MAC-counting utilities for executor discovery."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def _prod(values: Sequence[int]) -> int:
    """Product of an iterable of ints (1 for an empty iterable)."""
    result = 1
    for value in values:
        result *= int(value)
    return result


def _static_positive_shape(values: Optional[Sequence[int]]) -> Optional[List[int]]:
    """Normalize a shape to a static list of positive ints."""
    if not values:
        return None
    shape = [int(value) for value in values]
    if any(value <= 0 for value in shape):
        return None
    return shape


def _mac_detail(
    op_type: str,
    mac_count: int,
    formula: str,
    **fields: Any,
) -> Dict[str, Any]:
    """Build a standard MAC-detail record."""
    return {
        "op_type": op_type,
        "mac_count": int(mac_count),
        **fields,
        "formula": formula,
    }


def _conv_like_output_detail(
    *,
    op_type: str,
    output_shape: Sequence[int],
    weight_shape: Sequence[int],
    kernel_dims: Sequence[int],
    formula: str,
    input_dtype: Optional[str],
    weight_dtype: Optional[str],
    output_dtype: Optional[str],
) -> Dict[str, Any]:
    """Build a conv-like MAC record that scales with output volume."""
    per_element = _prod(kernel_dims)
    output_shape = [int(value) for value in output_shape]
    weight_shape = [int(value) for value in weight_shape]
    return _mac_detail(
        op_type,
        _prod(output_shape) * per_element,
        formula,
        output_shape=output_shape,
        weight_shape=weight_shape,
        per_element_macs=per_element,
        input_dtype=input_dtype,
        weight_dtype=weight_dtype,
        output_dtype=output_dtype,
    )


def _conv_like_input_detail(
    *,
    op_type: str,
    input_shape: Sequence[int],
    weight_shape: Sequence[int],
    kernel_dims: Sequence[int],
    formula: str,
    input_dtype: Optional[str],
    weight_dtype: Optional[str],
    output_dtype: Optional[str],
    output_shape: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Build a conv-like MAC record that scales with input volume."""
    per_element = _prod(kernel_dims)
    input_shape = [int(value) for value in input_shape]
    weight_shape = [int(value) for value in weight_shape]
    fields: Dict[str, Any] = {
        "input_shape": input_shape,
        "weight_shape": weight_shape,
        "per_element_macs": per_element,
        "input_dtype": input_dtype,
        "weight_dtype": weight_dtype,
        "output_dtype": output_dtype,
    }
    if output_shape is not None:
        fields["output_shape"] = [int(value) for value in output_shape]
    return _mac_detail(
        op_type,
        _prod(input_shape) * per_element,
        formula,
        **fields,
    )


def _matmul_output_detail(
    *,
    op_type: str,
    output_shape: Sequence[int],
    k: int,
    formula: str,
    output_dtype: Optional[str],
    shape_field_name: str,
    shape_field_value: Sequence[int],
    dtype_field_name: str,
    dtype_field_value: Optional[str],
) -> Dict[str, Any]:
    """Build a matmul-like MAC record that scales with output volume."""
    output_shape = [int(value) for value in output_shape]
    return _mac_detail(
        op_type,
        _prod(output_shape) * int(k),
        formula,
        output_shape=output_shape,
        K=int(k),
        output_dtype=output_dtype,
        **{
            shape_field_name: [int(value) for value in shape_field_value],
            dtype_field_name: dtype_field_value,
        },
    )


def _first_tensor_with_static_shape(
    tensors: Sequence[Dict[str, Any]],
    rank: int,
) -> tuple[Optional[Dict[str, Any]], Optional[List[int]]]:
    """Return the first tensor whose shape has the requested static rank."""
    for tensor in tensors:
        if len(tensor.get("shape", [])) != rank:
            continue
        shape = _static_positive_shape(tensor.get("shape"))
        if shape is not None:
            return tensor, shape
    return None, None


def _append_positive_mac_detail(
    nodes: List[Dict[str, Any]],
    detail: Optional[Dict[str, Any]],
    *,
    extra_fields: Optional[Dict[str, Any]] = None,
) -> int:
    """Append a MAC detail when it reports positive work and return its count."""
    if detail is None:
        return 0
    mac_count = int(detail.get("mac_count", 0))
    if mac_count <= 0:
        return 0
    if extra_fields:
        detail = {**extra_fields, **detail}
    nodes.append(detail)
    return mac_count


def _build_shape_map(graph) -> Dict[str, List[int]]:
    """Map tensor name -> static shape for a single-layer ONNX graph."""
    shapes: Dict[str, List[int]] = {}

    def _static_shape_from_vi(vi) -> Optional[List[int]]:
        tensor_type = vi.type.tensor_type
        if not tensor_type.shape.dim:
            return None
        dims: List[int] = []
        for dim in tensor_type.shape.dim:
            if dim.dim_value > 0:
                dims.append(dim.dim_value)
            else:
                return None
        return dims

    for collection in (graph.input, graph.output, graph.value_info):
        for value_info in collection:
            shape = _static_shape_from_vi(value_info)
            if shape is not None:
                shapes[value_info.name] = shape

    for initializer in graph.initializer:
        shapes[initializer.name] = [int(dim) for dim in initializer.dims]

    return shapes


_ONNX_DTYPE_NAMES = {
    1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16",
    6: "int32", 7: "int64", 8: "string", 9: "bool", 10: "float16",
    11: "float64", 12: "uint32", 13: "uint64", 14: "complex64",
    15: "complex128", 16: "bfloat16", 17: "float8e4m3fn",
    18: "float8e4m3fnuz", 19: "float8e5m2", 20: "float8e5m2fnuz",
    21: "uint4", 22: "int4",
}


def _dtype_name(elem_type: int) -> str:
    """Return a readable name for an ONNX element type."""
    return _ONNX_DTYPE_NAMES.get(int(elem_type), f"type_{int(elem_type)}")


def _build_dtype_map(graph) -> Dict[str, str]:
    """Map tensor name -> readable dtype for a single-layer ONNX graph."""
    dtypes: Dict[str, str] = {}

    for collection in (graph.input, graph.output, graph.value_info):
        for value_info in collection:
            elem_type = value_info.type.tensor_type.elem_type
            if elem_type:
                dtypes[value_info.name] = _dtype_name(elem_type)

    for initializer in graph.initializer:
        dtypes[initializer.name] = _dtype_name(initializer.data_type)

    return dtypes


def _tensor_dtype(name: str, dtypes: Dict[str, str]) -> Optional[str]:
    """Look up a tensor's dtype, returning ``None`` when unknown."""
    return dtypes.get(name)


def _get_attr_int(node, name: str, default: int) -> int:
    for attr in node.attribute:
        if attr.name == name:
            return int(attr.i)
    return default


def _get_attr_str(node, name: str, default: str = "") -> str:
    for attr in node.attribute:
        if attr.name == name:
            value = attr.s
            return value.decode() if isinstance(value, bytes) else str(value)
    return default


_CONV_WEIGHT_INPUT = {
    "Conv": 1,
    "ConvInteger": 1,
    "ConvTranspose": 1,
    "QLinearConv": 3,
}


_MATMUL_LHS_INPUT = {
    "MatMul": 0,
    "MatMulInteger": 0,
    "QLinearMatMul": 0,
}


def _tflite_mac_detail(layer: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Return MAC metadata for a single extracted TFLite layer, or ``None``."""
    op_type = layer.get("op_name")
    inputs = layer.get("inputs", [])
    outputs = layer.get("outputs", [])
    if not outputs:
        return None

    output_shape = _static_positive_shape(outputs[0].get("shape"))
    if output_shape is None:
        return None

    dynamic_inputs = [inp for inp in inputs if not inp.get("is_constant")]
    const_inputs = [inp for inp in inputs if inp.get("is_constant")]

    if op_type == "CONV_2D":
        weight, weight_shape = _first_tensor_with_static_shape(const_inputs, 4)
        if weight_shape is None:
            return None
        return _conv_like_output_detail(
            op_type=op_type,
            output_shape=output_shape,
            weight_shape=weight_shape,
            kernel_dims=weight_shape[1:],
            formula="prod(output_shape) * prod(weight_shape[1:])",
            input_dtype=dynamic_inputs[0].get("dtype") if dynamic_inputs else None,
            weight_dtype=weight.get("dtype"),
            output_dtype=outputs[0].get("dtype"),
        )

    if op_type == "DEPTHWISE_CONV_2D":
        weight, weight_shape = _first_tensor_with_static_shape(const_inputs, 4)
        if weight_shape is None:
            return None
        return _conv_like_output_detail(
            op_type=op_type,
            output_shape=output_shape,
            weight_shape=weight_shape,
            kernel_dims=weight_shape[1:3],
            formula="prod(output_shape) * prod(weight_shape[1:3])",
            input_dtype=dynamic_inputs[0].get("dtype") if dynamic_inputs else None,
            weight_dtype=weight.get("dtype"),
            output_dtype=outputs[0].get("dtype"),
        )

    if op_type == "TRANSPOSE_CONV":
        weight, weight_shape = _first_tensor_with_static_shape(const_inputs, 4)
        if weight is None or not dynamic_inputs:
            return None
        input_shape = _static_positive_shape(dynamic_inputs[0].get("shape"))
        if weight_shape is None or input_shape is None:
            return None
        return _conv_like_input_detail(
            op_type=op_type,
            input_shape=input_shape,
            output_shape=output_shape,
            weight_shape=weight_shape,
            kernel_dims=weight_shape[:3],
            formula="prod(input_shape) * prod(weight_shape[:3])",
            input_dtype=dynamic_inputs[0].get("dtype"),
            weight_dtype=weight.get("dtype"),
            output_dtype=outputs[0].get("dtype"),
        )

    if op_type == "FULLY_CONNECTED":
        weight, weight_shape = _first_tensor_with_static_shape(const_inputs, 2)
        if weight_shape is None:
            return None
        return _conv_like_output_detail(
            op_type=op_type,
            output_shape=output_shape,
            weight_shape=weight_shape,
            kernel_dims=[weight_shape[-1]],
            formula="prod(output_shape) * weight_shape[-1]",
            input_dtype=dynamic_inputs[0].get("dtype") if dynamic_inputs else None,
            weight_dtype=weight.get("dtype"),
            output_dtype=outputs[0].get("dtype"),
        )

    if op_type == "BATCH_MATMUL":
        lhs = dynamic_inputs[0] if dynamic_inputs else (inputs[0] if inputs else None)
        lhs_shape = _static_positive_shape(lhs.get("shape")) if lhs else None
        if lhs_shape is None:
            return None
        return _matmul_output_detail(
            op_type=op_type,
            output_shape=output_shape,
            k=int(lhs_shape[-1]),
            formula="prod(output_shape) * K",
            output_dtype=outputs[0].get("dtype"),
            shape_field_name="a_shape",
            shape_field_value=lhs_shape,
            dtype_field_name="input_dtype",
            dtype_field_value=lhs.get("dtype") if lhs else None,
        )

    return None


def compute_tflite_model_mac_details(model_path: Path) -> Dict[str, Any]:
    """Compute total MAC count plus per-op metadata for an extracted TFLite model."""
    from torq.lab.model_tools.extraction.tflite.layers import TFLiteLayerExtractor

    extractor = TFLiteLayerExtractor(str(model_path))
    nodes: List[Dict[str, Any]] = []
    total = 0
    for layer in extractor.get_layer_info():
        if layer.get("op_name") == "DELEGATE":
            continue
        total += _append_positive_mac_detail(nodes, _tflite_mac_detail(layer))
    return {"mac_count": total, "nodes": nodes}


def _conv_mac_detail(
    node, shapes: Dict[str, List[int]], dtypes: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    """MAC detail for convolution-family ops."""
    weight_idx = _CONV_WEIGHT_INPUT[node.op_type]
    if len(node.input) <= weight_idx or node.input[weight_idx] not in shapes:
        return None
    weight_name = node.input[weight_idx]
    weight_shape = shapes[weight_name]
    output_dtype = _tensor_dtype(node.output[0], dtypes) if node.output else None

    if node.op_type == "ConvTranspose":
        if not node.input or node.input[0] not in shapes:
            return None
        input_shape = shapes[node.input[0]]
        return _conv_like_input_detail(
            op_type=node.op_type,
            input_shape=input_shape,
            weight_shape=weight_shape,
            kernel_dims=weight_shape[1:],
            formula="prod(input_shape) * prod(weight_shape[1:])",
            input_dtype=_tensor_dtype(node.input[0], dtypes),
            weight_dtype=_tensor_dtype(weight_name, dtypes),
            output_dtype=output_dtype,
        )

    if not node.output or node.output[0] not in shapes:
        return None
    output_shape = shapes[node.output[0]]
    return _conv_like_output_detail(
        op_type=node.op_type,
        output_shape=output_shape,
        weight_shape=weight_shape,
        kernel_dims=weight_shape[1:],
        formula="prod(output_shape) * prod(weight_shape[1:])",
        input_dtype=_tensor_dtype(node.input[0], dtypes) if node.input else None,
        weight_dtype=_tensor_dtype(weight_name, dtypes),
        output_dtype=output_dtype,
    )


def _gemm_mac_detail(
    node, shapes: Dict[str, List[int]], dtypes: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    """MAC detail for a Gemm node (M x K x N)."""
    if len(node.input) < 2:
        return None
    a_name, b_name = node.input[0], node.input[1]
    if a_name not in shapes or b_name not in shapes:
        return None
    a_shape, b_shape = shapes[a_name], shapes[b_name]
    if len(a_shape) != 2 or len(b_shape) != 2:
        return None
    trans_a = _get_attr_int(node, "transA", 0)
    trans_b = _get_attr_int(node, "transB", 0)
    m = a_shape[1] if trans_a else a_shape[0]
    k = a_shape[0] if trans_a else a_shape[1]
    n = b_shape[0] if trans_b else b_shape[1]
    return _mac_detail(
        node.op_type,
        int(m) * int(k) * int(n),
        "M * K * N",
        a_shape=list(a_shape),
        b_shape=list(b_shape),
        trans_a=trans_a,
        trans_b=trans_b,
        M=int(m),
        K=int(k),
        N=int(n),
        a_dtype=_tensor_dtype(a_name, dtypes),
        b_dtype=_tensor_dtype(b_name, dtypes),
        output_dtype=_tensor_dtype(node.output[0], dtypes) if node.output else None,
    )


def _matmul_mac_detail(
    node, shapes: Dict[str, List[int]], dtypes: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    """MAC detail for matmul-family ops."""
    lhs_idx = _MATMUL_LHS_INPUT[node.op_type]
    if not node.output or node.output[0] not in shapes:
        return None
    if len(node.input) <= lhs_idx or node.input[lhs_idx] not in shapes:
        return None
    a_name = node.input[lhs_idx]
    a_shape = shapes[a_name]
    if len(a_shape) < 1:
        return None
    output_shape = shapes[node.output[0]]
    return _matmul_output_detail(
        op_type=node.op_type,
        output_shape=output_shape,
        k=int(a_shape[-1]),
        formula="prod(output_shape) * K",
        output_dtype=_tensor_dtype(node.output[0], dtypes),
        shape_field_name="a_shape",
        shape_field_value=a_shape,
        dtype_field_name="a_dtype",
        dtype_field_value=_tensor_dtype(a_name, dtypes),
    )


def _einsum_mac_detail(
    node, shapes: Dict[str, List[int]], dtypes: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    """MAC detail for an Einsum node."""
    equation = _get_attr_str(node, "equation").replace(" ", "")
    if not equation or "..." in equation:
        return None
    lhs = equation.split("->", 1)[0]
    operands = lhs.split(",")
    if len(operands) > len(node.input):
        return None

    label_size: Dict[str, int] = {}
    input_dtypes: List[Optional[str]] = []
    for operand, input_name in zip(operands, node.input):
        if input_name not in shapes:
            return None
        shape = shapes[input_name]
        if len(operand) != len(shape):
            return None
        for label, dim in zip(operand, shape):
            label_size[label] = int(dim)
        input_dtypes.append(_tensor_dtype(input_name, dtypes))

    if not label_size:
        return None
    return _mac_detail(
        node.op_type,
        _prod(label_size.values()),
        "prod(distinct index-label sizes)",
        equation=equation,
        label_sizes=dict(label_size),
        input_dtypes=input_dtypes,
        output_dtype=_tensor_dtype(node.output[0], dtypes) if node.output else None,
    )


def _rnn_mac_detail(
    node, shapes: Dict[str, List[int]], dtypes: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    """MAC detail for recurrent ops (RNN/GRU/LSTM)."""
    if len(node.input) < 3:
        return None
    x_name, w_name, r_name = node.input[0], node.input[1], node.input[2]
    if x_name not in shapes or w_name not in shapes or r_name not in shapes:
        return None
    x_shape = shapes[x_name]
    if len(x_shape) < 2:
        return None
    seq_length, batch = int(x_shape[0]), int(x_shape[1])
    w_shape, r_shape = shapes[w_name], shapes[r_name]
    w_numel, r_numel = _prod(w_shape), _prod(r_shape)
    return _mac_detail(
        node.op_type,
        seq_length * batch * (w_numel + r_numel),
        "seq_length * batch * (numel(W) + numel(R))",
        input_shape=list(x_shape),
        w_shape=list(w_shape),
        r_shape=list(r_shape),
        seq_length=seq_length,
        batch=batch,
        input_dtype=_tensor_dtype(x_name, dtypes),
        w_dtype=_tensor_dtype(w_name, dtypes),
        r_dtype=_tensor_dtype(r_name, dtypes),
    )


def _node_mac_detail(
    node, shapes: Dict[str, List[int]], dtypes: Dict[str, str]
) -> Optional[Dict[str, Any]]:
    """Return MAC metadata for a single ONNX node, or ``None``."""
    op_type = node.op_type

    if op_type in _CONV_WEIGHT_INPUT:
        return _conv_mac_detail(node, shapes, dtypes)
    if op_type == "Gemm":
        return _gemm_mac_detail(node, shapes, dtypes)
    if op_type in _MATMUL_LHS_INPUT:
        return _matmul_mac_detail(node, shapes, dtypes)
    if op_type == "Einsum":
        return _einsum_mac_detail(node, shapes, dtypes)
    if op_type in ("RNN", "GRU", "LSTM"):
        return _rnn_mac_detail(node, shapes, dtypes)
    return None


def _node_mac_count(node, shapes: Dict[str, List[int]], dtypes: Dict[str, str]) -> int:
    """Estimate multiply-accumulate operations for a single ONNX node."""
    detail = _node_mac_detail(node, shapes, dtypes)
    return detail.get("mac_count", 0) if detail else 0


def compute_model_mac_details(model) -> Dict[str, Any]:
    """Compute the total MAC count plus per-node metadata for an ONNX model."""
    graph = getattr(model, "graph", model)
    shapes = _build_shape_map(graph)
    dtypes = _build_dtype_map(graph)
    nodes: List[Dict[str, Any]] = []
    total = 0
    for node in graph.node:
        detail = _node_mac_detail(node, shapes, dtypes)
        total += _append_positive_mac_detail(
            nodes,
            detail,
            extra_fields={"node_name": node.name} if node.name else None,
        )
    return {"mac_count": total, "nodes": nodes}


def compute_model_mac_count(model) -> int:
    """Compute total multiply-accumulate operations for an ONNX model."""
    return compute_model_mac_details(model)["mac_count"]
