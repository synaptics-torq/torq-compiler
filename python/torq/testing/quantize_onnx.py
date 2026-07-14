# Copyright 2025-2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Static integer quantization for ONNX models.

This module provides a thin wrapper around ONNX Runtime's static quantization
and is shared between ``torq-gen-config`` and the pytest test suite. It is
focused on standard vision/classifier models with static input shapes.
"""

from pathlib import Path
import tempfile
from typing import Dict, List, Optional, Set, Tuple, Union
import copy

import numpy as np
import onnx
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)


# String values accepted for --quant-format.
_QUANT_FORMAT_MAP: Dict[str, QuantFormat] = {
    "qdq": QuantFormat.QDQ,
    "qoperator": QuantFormat.QOperator,
}

# Sentinel for the per-layer hybrid strategy (not a real ONNX Runtime enum).
_HYBRID_SENTINEL = "hybrid"

# String values accepted for --quant-dtype (case-insensitive).
# Maps to (activation_type, weight_type) tuples.
# Only A8W8 is supported today; more choices may be added once the backend
# supports mixed integer dtypes (e.g. A16W8).
_QUANT_DTYPE_MAP: Dict[str, Tuple[QuantType, QuantType]] = {
    "A8W8": (QuantType.QInt8, QuantType.QInt8),
}

# Ops that are known to work well in qoperator form and for which the TORQ
# backend has good support.  Per-layer hybrid quantization will pick QOperator
# for these and fall back to QDQ for everything else.
_QOPERATOR_PREFERRED_OPS: Set[str] = {
    "Conv",
    "Add",
    "Mul",
    "MatMul",
    "MaxPool",
    "AveragePool",
    "GlobalAveragePool",
    "Relu",
    "Clip",
    "LeakyRelu",
    "Sigmoid",
    "Softmax",
}


def _parse_quant_format(quant_format: str) -> QuantFormat:
    """Convert a string quant-format name to the ONNX Runtime enum."""
    fmt = quant_format.lower()
    if fmt == _HYBRID_SENTINEL:
        raise ValueError(
            "Use the hybrid path explicitly; _parse_quant_format does not map "
            "'hybrid' to a single ONNX Runtime QuantFormat."
        )
    if fmt not in _QUANT_FORMAT_MAP:
        raise ValueError(
            f"Unsupported quant-format: {quant_format!r}. "
            f"Supported values: {list(_QUANT_FORMAT_MAP.keys())}"
        )
    return _QUANT_FORMAT_MAP[fmt]


def _parse_quant_dtype(quant_dtype: str) -> Tuple[QuantType, QuantType]:
    """Convert a string quant-dtype name to (activation_type, weight_type)."""
    dtype = quant_dtype.upper()
    if dtype not in _QUANT_DTYPE_MAP:
        raise ValueError(
            f"Unsupported quant-dtype: {quant_dtype!r}. "
            f"Supported values: {list(_QUANT_DTYPE_MAP.keys())}"
        )
    return _QUANT_DTYPE_MAP[dtype]


def _compute_op_types(model: onnx.ModelProto) -> List[str]:
    """Return non-QDQ/Constant op types in the graph."""
    return [
        n.op_type
        for n in model.graph.node
        if n.op_type not in ("QuantizeLinear", "DequantizeLinear", "Constant")
    ]


def _resolve_hybrid_quant_format(model: onnx.ModelProto) -> QuantFormat:
    """Pick QOperator or QDQ for a single-layer or small subgraph model.

    ONNX Runtime's qoperator quantizer already emits a hybrid graph for full
    models (QLinear* for supported ops, DQ/Q islands for unsupported ones).  For
    per-layer extraction we can do even better by choosing the representation
    that is most likely to compile on the NSS slice for the ops in that layer.
    """
    compute_ops = _compute_op_types(model)
    if not compute_ops:
        return QuantFormat.QDQ
    # Use qoperator only when every compute op in the layer is in the preferred
    # set.  If any op is unsupported (e.g. ReduceMean) we fall back to QDQ for
    # the whole layer so ORT can insert DQ/Q islands around it.
    if all(op in _QOPERATOR_PREFERRED_OPS for op in compute_ops):
        return QuantFormat.QOperator
    return QuantFormat.QDQ


def _numpy_dtype(elem_type: int) -> np.dtype:
    """Map ONNX TensorProto element type to a numpy dtype."""
    mapping = {
        onnx.TensorProto.FLOAT: np.float32,
        onnx.TensorProto.DOUBLE: np.float64,
        onnx.TensorProto.INT64: np.int64,
        onnx.TensorProto.INT32: np.int32,
        onnx.TensorProto.INT16: np.int16,
        onnx.TensorProto.INT8: np.int8,
        onnx.TensorProto.UINT8: np.uint8,
        onnx.TensorProto.BOOL: bool,
    }
    if elem_type not in mapping:
        raise ValueError(f"Unsupported ONNX element type: {elem_type}")
    return mapping[elem_type]


def get_input_specs(model: onnx.ModelProto) -> List[Tuple[str, Tuple[int, ...], np.dtype]]:
    """Extract static input names, shapes, and dtypes from an ONNX model.

    Dynamic dimensions (dim_param) are replaced with 1 so that random
    calibration tensors can be generated.
    """
    specs = []
    for inp in model.graph.input:
        tensor_type = inp.type.tensor_type
        dtype = _numpy_dtype(tensor_type.elem_type)
        shape = []
        for dim in tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            else:
                shape.append(1)
        specs.append((inp.name, tuple(shape), dtype))
    return specs


class RandomCalibrationDataReader(CalibrationDataReader):
    """Produce deterministic random calibration tensors for static quantization."""

    def __init__(
        self,
        input_specs: List[Tuple[str, Tuple[int, ...], np.dtype]],
        num_samples: int = 20,
        seed: int = 0,
    ):
        self.input_specs = input_specs
        self.num_samples = num_samples
        self.rng = np.random.default_rng(seed)
        self.enum = iter(self._samples())

    def _samples(self):
        for _ in range(self.num_samples):
            yield {
                name: (self.rng.random(size=shape).astype(dtype) * 2.0 - 1.0)
                for name, shape, dtype in self.input_specs
            }

    def get_next(self):
        return next(self.enum, None)

    def reset(self):
        self.rng = np.random.default_rng(0)
        self.enum = iter(self._samples())


def quantize_onnx_static(
    model_input: Path,
    model_output: Path,
    *,
    num_calib: int = 20,
    quant_format: QuantFormat = QuantFormat.QDQ,
    activation_type: QuantType = QuantType.QInt8,
    weight_type: QuantType = QuantType.QInt8,
    per_channel: bool = False,
) -> Path:
    """Quantize an ONNX model to int8 using static calibration.

    Args:
        model_input: Path to the fp32 ONNX model.
        model_output: Path where the quantized ONNX model will be written.
        num_calib: Number of random calibration samples to use if no custom
            calibration reader is provided.
        quant_format: ONNX quantization format (QDQ or operator-oriented).
        activation_type: Activation quantization type (QInt8 or QUInt8).
        weight_type: Weight quantization type (QInt8 or QUInt8).
        per_channel: Whether to use per-channel weight scales.

    Returns:
        The path to the written quantized model.
    """
    model = onnx.load(str(model_input))
    model_output = Path(model_output)
    model_output.parent.mkdir(parents=True, exist_ok=True)

    input_specs = get_input_specs(model)
    if not input_specs:
        raise ValueError(f"Model {model_input} has no graph inputs")
    calib_data_reader = RandomCalibrationDataReader(input_specs, num_samples=num_calib)

    quantize_static(
        model_input=str(model_input),
        model_output=str(model_output),
        calibration_data_reader=calib_data_reader,
        quant_format=quant_format,
        activation_type=activation_type,
        weight_type=weight_type,
        per_channel=per_channel,
    )
    return model_output


def quantize_onnx_static_from_model(
    model: onnx.ModelProto,
    model_output: Path,
    *,
    num_calib: int = 20,
    quant_format: QuantFormat = QuantFormat.QDQ,
    activation_type: QuantType = QuantType.QInt8,
    weight_type: QuantType = QuantType.QInt8,
    per_channel: bool = False,
) -> onnx.ModelProto:
    """Quantize an in-memory ONNX model and return the quantized model.

    Writes the quantized model to *model_output* and loads it back so callers
    can continue processing the model object directly.
    """
    model_output = Path(model_output)
    model_output.parent.mkdir(parents=True, exist_ok=True)

    # Resolve the per-layer hybrid strategy before we call ORT.
    if quant_format == _HYBRID_SENTINEL:
        quant_format = _resolve_hybrid_quant_format(model)

    input_specs = get_input_specs(model)
    if not input_specs:
        raise ValueError("Model has no graph inputs")
    calib_data_reader = RandomCalibrationDataReader(input_specs, num_samples=num_calib)

    quantize_static(
        model_input=model,
        model_output=str(model_output),
        calibration_data_reader=calib_data_reader,
        quant_format=quant_format,
        activation_type=activation_type,
        weight_type=weight_type,
        per_channel=per_channel,
    )
    return onnx.load(str(model_output))


def convert_qdq_to_full_integer(
    model: onnx.ModelProto, io_dtype: int = onnx.TensorProto.INT8
) -> onnx.ModelProto:
    """Rewrite a QDQ model so its I/O tensors are the requested integer dtype.

    ONNX Runtime's static quantization keeps graph inputs/outputs as float32.
    This helper removes the input-side QuantizeLinear and the output-side
    DequantizeLinear so the model accepts and returns integer tensors directly,
    matching the deployment-style integer I/O used by quantized models.

    Args:
        model: Quantized ONNX model in QDQ format.
        io_dtype: Target ONNX TensorProto element type for graph I/O (default: INT8).
    """
    model = copy.deepcopy(model)
    graph = model.graph

    input_names = {inp.name for inp in graph.input}
    output_names = {out.name for out in graph.output}

    # Build helper maps.
    node_by_output: Dict[str, onnx.NodeProto] = {}
    consumers_by_output: Dict[str, List[onnx.NodeProto]] = {}
    for node in graph.node:
        for out in node.output:
            node_by_output[out] = node
        for inp in node.input:
            consumers_by_output.setdefault(inp, []).append(node)

    # Helper to get shape from value_info / input / output / initializer.
    def get_shape(name: str) -> Optional[List[int]]:
        for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
            if vi.name == name:
                if vi.type.HasField("tensor_type"):
                    return [
                        d.dim_value if d.HasField("dim_value") else -1
                        for d in vi.type.tensor_type.shape.dim
                    ]
        for init in graph.initializer:
            if init.name == name:
                return list(init.dims)
        return None

    nodes_to_remove: Set[int] = set()

    # 1) Convert inputs: remove QuantizeLinear nodes that consume graph inputs.
    for inp in list(graph.input):
        in_name = inp.name
        consumers = consumers_by_output.get(in_name, [])
        quantize_nodes = [n for n in consumers if n.op_type == "QuantizeLinear"]
        if not quantize_nodes:
            continue
        # Only safe if every consumer is a QuantizeLinear.
        if len(quantize_nodes) != len(consumers):
            continue

        for q_node in quantize_nodes:
            q_out = q_node.output[0]
            # Promote q_out to be the graph input (integer).
            shape = get_shape(q_out) or get_shape(in_name)
            if shape is None:
                raise ValueError(f"Cannot determine shape for quantized input {q_out}")
            new_input = onnx.helper.make_tensor_value_info(q_out, io_dtype, shape)
            graph.input.remove(inp)
            graph.input.append(new_input)

            # All consumers of q_out now use the graph input directly.
            nodes_to_remove.add(id(q_node))

    # 2) Convert outputs: remove DequantizeLinear nodes that produce graph outputs.
    for out in list(graph.output):
        out_name = out.name
        producer = node_by_output.get(out_name)
        if producer is None or producer.op_type != "DequantizeLinear":
            continue

        dq_in = producer.input[0]
        # Promote dq_in to be the graph output (integer).
        shape = get_shape(dq_in) or get_shape(out_name)
        if shape is None:
            raise ValueError(f"Cannot determine shape for quantized output {dq_in}")
        new_output = onnx.helper.make_tensor_value_info(dq_in, io_dtype, shape)
        graph.output.remove(out)
        graph.output.append(new_output)

        nodes_to_remove.add(id(producer))

    # Rebuild node list without removed nodes.
    new_nodes = [n for n in graph.node if id(n) not in nodes_to_remove]
    del graph.node[:]
    graph.node.extend(new_nodes)

    # Refresh value_info and run shape inference.
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass

    return model


def is_model_quantized(model: onnx.ModelProto) -> bool:
    """Return True if the model already contains quantization nodes or int8/int16 tensors."""
    integer_types = {
        onnx.TensorProto.INT8,
        onnx.TensorProto.UINT8,
        onnx.TensorProto.INT16,
        onnx.TensorProto.UINT16,
    }
    for node in model.graph.node:
        if node.op_type in ("QuantizeLinear", "DequantizeLinear"):
            return True
    for init in model.graph.initializer:
        if init.data_type in integer_types:
            return True
    for value_info in list(model.graph.input) + list(model.graph.output):
        if value_info.type.tensor_type.elem_type in integer_types:
            return True
    return False


def quantize_onnx_model(
    model: onnx.ModelProto,
    *,
    num_calib: int = 20,
    per_channel: bool = False,
    full_integer: bool = False,
    quant_format: str = "qdq",
    quant_dtype: str = "A8W8",
) -> onnx.ModelProto:
    """Quantize an in-memory ONNX model to integer and return the quantized model.

    Args:
        model: The FP32 ONNX model to quantize.
        num_calib: Number of random calibration samples.
        per_channel: Whether to use per-channel weight quantization.
        full_integer: If True, rewrite graph I/O to integer by stripping the
            input QuantizeLinear and output DequantizeLinear nodes.
        quant_format: ONNX quantization format, "qdq" (default), "qoperator",
            or "hybrid".  "hybrid" resolves per-layer to either QDQ or
            QOperator based on the ops present in the model.
        quant_dtype: Activation/weight quantization dtype string, e.g. "A8W8".

    Returns:
        The quantized ONNX model.
    """
    if is_model_quantized(model):
        return model

    activation_type, weight_type = _parse_quant_dtype(quant_dtype)

    if quant_format == _HYBRID_SENTINEL:
        onnx_quant_format = _resolve_hybrid_quant_format(model)
    else:
        onnx_quant_format = _parse_quant_format(quant_format)

    with tempfile.TemporaryDirectory() as tmp:
        qdq_path = Path(tmp) / "qdq.onnx"
        quantize_onnx_static_from_model(
            model,
            qdq_path,
            num_calib=num_calib,
            per_channel=per_channel,
            quant_format=onnx_quant_format,
            activation_type=activation_type,
            weight_type=weight_type,
        )
        quantized = onnx.load(str(qdq_path))

    if full_integer:
        quantized = convert_qdq_to_full_integer(
            quantized, io_dtype=activation_type.tensor_type
        )

    return quantized


# Common option specs shared by the CLI and pytest so the same flags are
# accepted everywhere. Each entry is (name, kwargs_for_add_argument).
_ONNX_QUANTIZATION_OPTIONS = [
    (
        "--quantize",
        {
            "action": "store_true",
            "default": False,
            "help": "Quantize ONNX models/layers to int8 before testing",
        },
    ),
    (
        "--per-channel",
        {
            "action": "store_true",
            "default": False,
            "help": "Use per-channel weight quantization with --quantize",
        },
    ),
    (
        "--full-integer",
        {
            "action": "store_true",
            "default": False,
            "help": "Rewrite quantized model I/O to int8 (remove input Q/output DQ)",
        },
    ),
    (
        "--quant-format",
        {
            "default": "qdq",
            "choices": ["qdq", "qoperator", "hybrid"],
            "help": (
                "ONNX quantization format: qdq (default), qoperator, or hybrid. "
                "Hybrid resolves per-layer to QDQ or QOperator based on the ops "
                "present."
            ),
        },
    ),
    (
        "--quant-dtype",
        {
            "default": "A8W8",
            "choices": ["A8W8"],
            "help": (
                "Quantized integer dtype for activations and weights. "
                "A8W8 (default) uses signed 8-bit for both."
            ),
        },
    ),
]


def add_onnx_quantization_args(parser):
    """Add the shared ONNX quantization flags to an ``argparse.ArgumentParser``."""
    for name, kwargs in _ONNX_QUANTIZATION_OPTIONS:
        parser.add_argument(name, **kwargs)
    return parser


def add_onnx_quantization_options(parser):
    """Add the shared ONNX quantization flags to a pytest ``Parser``."""
    for name, kwargs in _ONNX_QUANTIZATION_OPTIONS:
        parser.addoption(name, **kwargs)
    return parser
