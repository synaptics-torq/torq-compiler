# Copyright 2025-2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Static integer quantization for ONNX models.

This module provides a thin wrapper around ONNX Runtime's static quantization
and is shared between ``torq-gen-config`` and the in-tree test suite. It is
focused on standard vision/classifier models with static input shapes.
"""

import argparse
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

from torq.lab.logging import add_logging_args, configure_logging

__all__ = [
    "CalibrationDataReader",
    "ONNX_STATIC_QUANTIZATION_OPTIONS",
    "QuantFormat",
    "QuantType",
    "RandomCalibrationDataReader",
    "add_onnx_static_quantization_args",
    "add_onnx_static_quantization_options",
    "add_static_quant_flags",
    "add_static_quantize_args",
    "convert_qdq_to_full_integer",
    "get_input_specs",
    "is_model_quantized",
    "onnx_static_quantize",
    "onnx_static_quantize_file",
    "parse_quant_dtype",
    "parse_quant_format",
    "quantize_static",
    "static_quantize_from_args",
]


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


def parse_quant_format(quant_format: str) -> QuantFormat:
    """Convert a string quant-format name to the ONNX Runtime enum."""
    fmt = quant_format.lower()
    if fmt == _HYBRID_SENTINEL:
        raise ValueError(
            "Use the hybrid path explicitly; parse_quant_format does not map "
            "'hybrid' to a single ONNX Runtime QuantFormat."
        )
    if fmt not in _QUANT_FORMAT_MAP:
        raise ValueError(
            f"Unsupported quant-format: {quant_format!r}. "
            f"Supported values: {list(_QUANT_FORMAT_MAP.keys())}"
        )
    return _QUANT_FORMAT_MAP[fmt]


def parse_quant_dtype(quant_dtype: str) -> Tuple[QuantType, QuantType]:
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


# Shared message for the dataset entry points, which are placeholders until
# canonical dataset support lands.
_DATASET_NOT_IMPLEMENTED_MSG = (
    "Static ONNX quantization dataset support is not implemented yet, check back soon"
)


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


def _validate_node_and_op_selection(
    model: onnx.ModelProto,
    quantize_only_nodes: Optional[List[str]],
    exclude_nodes: Optional[List[str]],
    quantize_only_ops: Optional[List[str]],
) -> None:
    """Reject selection names that match no node/op in the graph.

    ONNX Runtime's quantizer silently ignores unknown node/op names, which
    would produce a fully float "quantized" model with no warning.
    """
    node_names = {node.name for node in model.graph.node}
    for flag, names in (
        ("--quantize-only-nodes", quantize_only_nodes),
        ("--exclude-nodes", exclude_nodes),
    ):
        unknown = [name for name in (names or []) if name not in node_names]
        if unknown:
            raise ValueError(
                f"{flag}: unknown node name(s) {unknown}; not found among the "
                f"model's {len(node_names)} nodes"
            )
    op_types = {node.op_type for node in model.graph.node}
    unknown = [op for op in (quantize_only_ops or []) if op not in op_types]
    if unknown:
        raise ValueError(
            f"--quantize-only-ops: op type(s) {unknown} not present in the model; "
            f"op types present: {sorted(op_types)}"
        )


def onnx_static_quantize(
    model: onnx.ModelProto,
    *,
    num_calib: int = 20,
    calibration_data_reader: Optional[CalibrationDataReader] = None,
    per_channel: bool = False,
    full_integer: bool = False,
    quant_format: str = "qdq",
    quant_dtype: str = "A8W8",
    quantize_only_ops: Optional[List[str]] = None,
    quantize_only_nodes: Optional[List[str]] = None,
    exclude_nodes: Optional[List[str]] = None,
) -> onnx.ModelProto:
    """Quantize an in-memory ONNX model to integer and return the quantized model.

    Thin ModelProto wrapper around ONNX Runtime's ``quantize_static``; the
    onnxruntime pass runs against a scratch file. File-path I/O is provided
    by :func:`onnx_static_quantize_file`.

    Args:
        model: The FP32 ONNX model to quantize.
        num_calib: Number of random calibration samples when
            calibration_data_reader is not provided.
        calibration_data_reader: ONNX Runtime calibration reader that yields
            representative model-input dictionaries. If omitted, deterministic
            synthetic random data is used.
        per_channel: Whether to use per-channel weight quantization.
        full_integer: If True, rewrite graph I/O to integer by stripping the
            input QuantizeLinear and output DequantizeLinear nodes.
        quant_format: ONNX quantization format, "qdq" (default), "qoperator",
            or "hybrid".  "hybrid" resolves per-layer to either QDQ or
            QOperator based on the ops present in the model.
        quant_dtype: Activation/weight quantization dtype string, e.g. "A8W8".
        quantize_only_ops: Only quantize the given ONNX op types.
        quantize_only_nodes: Only quantize the given node names.
        exclude_nodes: Exclude the given node names from quantization (e.g.
            an analysis exclude list).

    Returns:
        The quantized ONNX model.
    """
    if is_model_quantized(model):
        raise ValueError(
            "Model appears to be already quantized (contains QuantizeLinear/"
            "DequantizeLinear nodes or integer tensors); re-quantization is "
            "not supported. Quantize the fp32 source model instead."
        )
    _validate_node_and_op_selection(
        model, quantize_only_nodes, exclude_nodes, quantize_only_ops
    )

    activation_type, weight_type = parse_quant_dtype(quant_dtype)

    if quant_format == _HYBRID_SENTINEL:
        onnx_quant_format = _resolve_hybrid_quant_format(model)
    else:
        onnx_quant_format = parse_quant_format(quant_format)

    with tempfile.TemporaryDirectory() as tmp:
        qdq_path = Path(tmp) / "qdq.onnx"
        if calibration_data_reader is None:
            input_specs = get_input_specs(model)
            if not input_specs:
                raise ValueError("Model has no graph inputs")
            calibration_data_reader = RandomCalibrationDataReader(
                input_specs, num_samples=num_calib
            )
        quantize_static(
            model_input=model,
            model_output=str(qdq_path),
            calibration_data_reader=calibration_data_reader,
            quant_format=onnx_quant_format,
            activation_type=activation_type,
            weight_type=weight_type,
            per_channel=per_channel,
            op_types_to_quantize=quantize_only_ops,
            nodes_to_quantize=quantize_only_nodes,
            nodes_to_exclude=exclude_nodes,
        )
        quantized = onnx.load(str(qdq_path))

    if full_integer:
        quantized = convert_qdq_to_full_integer(
            quantized, io_dtype=activation_type.tensor_type
        )

    return quantized


def onnx_static_quantize_file(
    model_input,
    model_output,
    *,
    num_calib: int = 20,
    calibration_data_reader: Optional[CalibrationDataReader] = None,
    per_channel: bool = False,
    full_integer: bool = False,
    quant_format: str = "qdq",
    quant_dtype: str = "A8W8",
    quantize_only_ops: Optional[List[str]] = None,
    quantize_only_nodes: Optional[List[str]] = None,
    exclude_nodes: Optional[List[str]] = None,
    dataset: Optional[Union[str, Path]] = None,
) -> Path:
    """Quantize an ONNX model file to integer using static calibration.

    Thin file wrapper around :func:`onnx_static_quantize`: loads *model_input*,
    runs the static (calibration-based) flow, and writes the quantized model
    to *model_output*.

    Args:
        model_input: Path to the fp32 ONNX model.
        model_output: Path where the quantized ONNX model will be written.
        num_calib: Number of random calibration samples when
            calibration_data_reader is not provided.
        calibration_data_reader: ONNX Runtime calibration reader that yields
            representative model-input dictionaries. If omitted, deterministic
            synthetic random data is used.
        per_channel: Whether to use per-channel weight quantization.
        full_integer: If True, rewrite graph I/O to integer by stripping the
            input QuantizeLinear and output DequantizeLinear nodes.
        quant_format: ONNX quantization format, "qdq" (default), "qoperator",
            or "hybrid".  "hybrid" resolves per-layer to either QDQ or
            QOperator based on the ops present in the model.
        quant_dtype: Activation/weight quantization dtype string, e.g. "A8W8".
        quantize_only_ops: Only quantize the given ONNX op types.
        quantize_only_nodes: Only quantize the given node names.
        exclude_nodes: Exclude the given node names from quantization (e.g.
            an analysis exclude list).
        dataset: Placeholder for canonical dataset support, which is not
            implemented yet; passing a value raises NotImplementedError.

    Returns:
        The path to the written quantized model.
    """
    if dataset is not None:
        raise NotImplementedError(_DATASET_NOT_IMPLEMENTED_MSG)

    model_input = Path(model_input)
    model_output = Path(model_output)

    model = onnx.load(str(model_input))
    quantized = onnx_static_quantize(
        model,
        num_calib=num_calib,
        calibration_data_reader=calibration_data_reader,
        per_channel=per_channel,
        full_integer=full_integer,
        quant_format=quant_format,
        quant_dtype=quant_dtype,
        quantize_only_ops=quantize_only_ops,
        quantize_only_nodes=quantize_only_nodes,
        exclude_nodes=exclude_nodes,
    )
    model_output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(quantized, str(model_output))
    return model_output


# Flag specs (name, kwargs_for_add_argument) for the static quantization format
# options accepted by every CLI surface (``torq-lab quantize static``,
# ``torq-gen-config quantize``, gen-config discover/run, and the ``--quantize``
# pytest options), so the same flags and help text are used everywhere.
_STATIC_QUANT_FLAGS = [
    (
        "--per-channel",
        {
            "action": "store_true",
            "default": False,
            "help": "Use per-channel weight quantization",
        },
    ),
    (
        "--full-integer",
        {
            "action": "store_true",
            "default": False,
            "help": "Rewrite I/O to integer (remove input Q and output DQ nodes)",
        },
    ),
    (
        "--quant-format",
        {
            "default": "qdq",
            "choices": ["qdq", "qoperator", "hybrid"],
            "help": (
                "ONNX quantization format: qdq (default), qoperator, or hybrid. "
                "Hybrid picks qoperator for ops with good qoperator support "
                "(Conv, Add, MatMul, ...) and qdq for the rest, applied per layer."
            ),
        },
    ),
    (
        "--quant-dtype",
        {
            "default": "A8W8",
            "choices": ["A8W8"],
            "help": (
                "Quantized activation/weight integer dtype combination "
                "(case-insensitive). Currently only A8W8 is supported."
            ),
        },
    ),
]


def add_static_quant_flags(parser, skip: Tuple[str, ...] = ()):
    """Add the shared static quantization format flags to an argument parser.

    ``skip`` lists flag names to omit (e.g. ``--full-integer`` for sensitivity
    analysis, where integer I/O would break output comparison).
    """
    for name, kwargs in _STATIC_QUANT_FLAGS:
        if name in skip:
            continue
        parser.add_argument(name, **kwargs)
    return parser


# Common option specs shared by the CLI and the test suite so the same flags
# are accepted everywhere. Each entry is (name, kwargs_for_add_argument).
ONNX_STATIC_QUANTIZATION_OPTIONS = [
    (
        "--quantize",
        {
            "action": "store_true",
            "default": False,
            "help": "Quantize ONNX models/layers to int8 before testing",
        },
    ),
] + _STATIC_QUANT_FLAGS


def add_onnx_static_quantization_args(parser):
    """Add the shared ONNX static quantization flags to an ``argparse.ArgumentParser``."""
    for name, kwargs in ONNX_STATIC_QUANTIZATION_OPTIONS:
        parser.add_argument(name, **kwargs)
    return parser


def add_onnx_static_quantization_options(parser):
    """Add the shared ONNX static quantization flags to an argument parser."""
    for name, kwargs in ONNX_STATIC_QUANTIZATION_OPTIONS:
        parser.addoption(name, **kwargs)
    return parser


def add_static_quantize_args(parser: argparse.ArgumentParser) -> None:
    """Add the ``torq-lab quantize static`` / ``torq-quantize-model static`` flags."""
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input fp32 ONNX model path",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Output quantized ONNX model path (default: <input-stem>.int8.onnx)",
    )
    parser.add_argument(
        "--num-calib",
        type=int,
        default=20,
        help="Number of synthetic calibration samples (default: 20)",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Calibration dataset (not implemented yet, check back soon)",
    )
    add_static_quant_flags(parser)
    parser.add_argument(
        "--quantize-only-ops",
        type=str,
        nargs="+",
        default=None,
        help="Only quantize specified op types; must be valid ONNX op types",
    )
    parser.add_argument(
        "--quantize-only-nodes",
        type=str,
        nargs="+",
        default=None,
        help="Only quantize specified nodes; must be valid node names from graph",
    )
    parser.add_argument(
        "--exclude-nodes",
        type=str,
        nargs="+",
        default=None,
        help="Exclude specified nodes from quantization (e.g. an analysis exclude list)",
    )
    add_logging_args(parser)


def static_quantize_from_args(args: argparse.Namespace) -> Path:
    """Run static quantization from CLI args; returns the written model path."""
    configure_logging(args.logging)
    model_path = Path(args.input)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    output_path = Path(args.output) if args.output else model_path.with_suffix(".int8.onnx")
    return onnx_static_quantize_file(
        model_path,
        output_path,
        num_calib=args.num_calib,
        dataset=args.dataset,
        per_channel=args.per_channel,
        full_integer=args.full_integer,
        quant_format=args.quant_format,
        quant_dtype=args.quant_dtype,
        quantize_only_ops=args.quantize_only_ops,
        quantize_only_nodes=args.quantize_only_nodes,
        exclude_nodes=args.exclude_nodes,
    )
