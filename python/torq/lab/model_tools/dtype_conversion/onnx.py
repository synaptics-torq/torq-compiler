# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright © 2025 Synaptics Incorporated.

import os
import argparse
import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from types import EllipsisType
from typing import Final, Union

import numpy as np
import onnx
import onnx_graphsurgeon as gs


__all__ = [
    "OnnxDtypeConverterBase",
    "FP32Converter",
    "Int64Converter",
    "convert_onnx_model",
    "convert_model",
    "add_onnx_dtype_convert_args",
    "onnx_dtype_convert_from_args",
]

logger = logging.getLogger("ONNX-Dtype-Converter")

_ONNX_DTYPE_MAPPING: Final[dict[str, onnx.TensorProto.DataType]] = {
    "fp32": onnx.TensorProto.FLOAT,
    "bf16": onnx.TensorProto.BFLOAT16,
    "fp16": onnx.TensorProto.FLOAT16,
    "int64": onnx.TensorProto.INT64,
    "uint64": onnx.TensorProto.UINT64,
    "int32": onnx.TensorProto.INT32,
    "uint32": onnx.TensorProto.UINT32,
    "int16": onnx.TensorProto.INT16,
    "uint16": onnx.TensorProto.UINT16,
    "int8": onnx.TensorProto.INT8,
    "uint8": onnx.TensorProto.UINT8,
}

_EnforcedIoIndices = tuple[int | EllipsisType, ...]
_EnforcedIoSpec = tuple[_EnforcedIoIndices, ...]

_ONNX_TENSOR_TYPE_STRINGS: Final[dict[onnx.TensorProto.DataType, str]] = {
    onnx.TensorProto.BFLOAT16: "tensor(bfloat16)",
    onnx.TensorProto.FLOAT: "tensor(float)",
    onnx.TensorProto.FLOAT16: "tensor(float16)",
    onnx.TensorProto.INT64: "tensor(int64)",
    onnx.TensorProto.UINT64: "tensor(uint64)",
    onnx.TensorProto.INT32: "tensor(int32)",
    onnx.TensorProto.UINT32: "tensor(uint32)",
    onnx.TensorProto.INT16: "tensor(int16)",
    onnx.TensorProto.UINT16: "tensor(uint16)",
    onnx.TensorProto.INT8: "tensor(int8)",
    onnx.TensorProto.UINT8: "tensor(uint8)",
}


# -----------------------------------------------------------------------------
# Private conversion helpers
#
# Inlined from the old ``utils/onnx.py`` (their only consumer was this
# converter). They keep logging under the historical ``torq.lab.utils.onnx``
# logger name so the ``torq-convert-dtype`` CLI's stderr output is unchanged.
# -----------------------------------------------------------------------------

_utils_logger = logging.getLogger("torq.lab.utils.onnx")


# -----------------------------------------------------------------------------
# Model inspection utilities
# -----------------------------------------------------------------------------

def _get_model_opset(
    model: str | os.PathLike | onnx.ModelProto,
    opset_domains: list[str] = ["ai.onnx", ""],
    default_opset: int = 17
) -> int:
    if not isinstance(model, onnx.ModelProto):
        model = onnx.load(model)
    for opset_id in model.opset_import:
        if opset_id.domain in opset_domains:
            return int(opset_id.version)
    _utils_logger.warning("Cannot determine opset for model, defaulting to %d", default_opset)
    return default_opset


# -----------------------------------------------------------------------------
# DType utilities
# -----------------------------------------------------------------------------

_DTypeLike = Union[int, np.dtype, type, str, None]

def _is_same_dtype(typ1: _DTypeLike, typ2: _DTypeLike) -> bool:
    if typ1 is typ2:
        return True
    if typ1 == typ2:
        return True

    def _to_np_dtype(typ: _DTypeLike) -> np.dtype | None:
        if typ is None:
            return None
        if isinstance(typ, np.dtype):
            return typ
        if isinstance(typ, int):
            try:
                return np.dtype(onnx.helper.tensor_dtype_to_np_dtype(typ))
            except (TypeError, ValueError, KeyError):
                return None
        try:
            return np.dtype(typ)
        except TypeError:
            return None

    dt1 = _to_np_dtype(typ1)
    dt2 = _to_np_dtype(typ2)
    return dt1 is not None and dt2 is not None and dt1 == dt2


# -----------------------------------------------------------------------------
# Transformations
# -----------------------------------------------------------------------------

def _drop_empty_name_value_info(model: onnx.ModelProto) -> onnx.ModelProto:
    """Remove ``graph.value_info`` entries with an empty name.

    Rare leftover when an RNN optional output still uses ``""``; fails ONNX
    validation: ``Field 'name' of 'value_info' is required to be non-empty``.
    """
    graph = model.graph
    kept = [vi for vi in graph.value_info if vi.name]
    if len(kept) != len(graph.value_info):
        del graph.value_info[:]
        graph.value_info.extend(kept)
    return model


def _finalize_torq_ready_onnx(
    model: onnx.ModelProto,
    *,
    max_ir_version: int = 11,
    symbolic_shape_infer: bool = True,
) -> onnx.ModelProto:
    """Post-process ONNX for Torq import and layer extraction.

    - Optionally runs ONNX Runtime symbolic shape inference so ``unk__`` dims
      become static where ORT can derive them.
    - Drops ``graph.value_info`` entries whose names duplicate ``graph.output``
      (avoids rank mismatches in torch-onnx import for isolated subgraphs).
    - Caps ``ir_version`` for broader onnxruntime / tooling compatibility.
    - Refreshes standard ONNX shape inference when possible.

    Mutates ``model`` unless a shape-inference step returns a replacement
    model. Requires ``onnxruntime`` with ``tools.symbolic_shape_infer`` for the
    first step; if unavailable or it fails, subsequent steps still run.
    """
    work = model
    if symbolic_shape_infer:
        try:
            from onnxruntime.tools.symbolic_shape_infer import (
                SymbolicShapeInference,
            )

            work = SymbolicShapeInference.infer_shapes(
                work,
                auto_merge=True,
                guess_output_rank=True,
                verbose=0,
            )
        except Exception as exc:
            _utils_logger.debug("Symbolic shape inference skipped: %s", exc)

    work.ir_version = min(int(work.ir_version), max_ir_version)

    work = _drop_empty_name_value_info(work)

    graph = work.graph
    out_names = {o.name for o in graph.output}
    kept = [vi for vi in graph.value_info if vi.name and vi.name not in out_names]
    del graph.value_info[:]
    graph.value_info.extend(kept)

    try:
        work = onnx.shape_inference.infer_shapes(work)
    except Exception as exc:
        _utils_logger.debug("shape_inference.infer_shapes after finalize skipped: %s", exc)

    try:
        onnx.checker.check_model(work, full_check=False)
    except Exception as exc:
        _utils_logger.warning("ONNX checker warning after finalize_torq_ready_onnx: %s", exc)

    return work


def _upgrade_model(model: onnx.ModelProto, target_opset: int) -> onnx.ModelProto:
    if (curr_opset := _get_model_opset(model)) >= target_opset:
        _utils_logger.info("Model already at opset %d >= %d, skipping upgrade", curr_opset, target_opset)
        return model
    upgraded = onnx.version_converter.convert_version(model, target_opset)
    _utils_logger.info("Upgraded model opset to %d", target_opset)
    return upgraded

class OnnxDtypeConverterBase(ABC):

    _ALL_IO_INDICES: Final = Ellipsis
    _ENFORCED_IO: Final[dict[str, _EnforcedIoSpec]] = {}

    def __init__(
        self,
        original_dtype: str,
        export_dtype: str,
        convert_io: bool = False,
        direct_cast: bool = True,
        remove_unused_node_outputs: bool = False,
        enforce_io_casts: bool = False,
    ):
        if export_dtype not in self.allowed_dtypes():
            raise ValueError(
                f"Invalid export dtype '{export_dtype}', select from {sorted(self.allowed_dtypes())}"
            )
        self._original_dtype_str = original_dtype
        self._export_dtype_str = export_dtype
        self._convert_io = convert_io
        self._direct_cast = direct_cast
        self._enforce_io_casts = enforce_io_casts

        self._original_onnx_dtype = self._validate_dtype(original_dtype)
        self._export_onnx_dtype = self._validate_dtype(export_dtype)
        self._convert_exceptions: dict[str, str] = {}
        self._remove_unused_node_outputs = remove_unused_node_outputs
        self._opset_version: int | None = None

    @staticmethod
    def _validate_dtype(dtype: str) -> onnx.TensorProto.DataType:
        onnx_dtype = _ONNX_DTYPE_MAPPING.get(dtype)
        if onnx_dtype is None:
            raise ValueError(
                f"Invalid dtype '{dtype}', select from {list(_ONNX_DTYPE_MAPPING.keys())}"
            )
        return onnx_dtype

    @staticmethod
    def _collect_all_graphs(root: gs.Graph) -> list[gs.Graph]:
        graphs: list[gs.Graph] = []
        queue: list[gs.Graph] = [root]

        while queue:
            g = queue.pop()
            graphs.append(g)
            for node in g.nodes:
                for attr_val in node.attrs.values():
                    if isinstance(attr_val, gs.Graph):
                        queue.append(attr_val)
                    elif isinstance(attr_val, list):
                        for v in attr_val:
                            if isinstance(v, gs.Graph):
                                queue.append(v)

        return graphs

    def _is_original_dtype(self, dtype) -> bool:
        return _is_same_dtype(dtype, self._original_onnx_dtype)

    def _get_enforced_io(self, node: gs.Node) -> _EnforcedIoSpec | None:
        # The ONNX schema is the source of truth for fixed-type operator I/O.
        # This handles every supported export dtype instead of maintaining an
        # incomplete manual operator list.
        schema_enforced = self._get_schema_enforced_io(node)
        explicit_enforced = self._ENFORCED_IO.get(node.op)
        if schema_enforced is None:
            return explicit_enforced
        if explicit_enforced is None:
            return schema_enforced
        return (
            tuple(sorted(set(schema_enforced[0]) | set(explicit_enforced[0]))),
            tuple(sorted(set(schema_enforced[1]) | set(explicit_enforced[1]))),
        )

    @staticmethod
    def _schema_type_options(schema, formal) -> tuple[str, ...]:
        if formal.type_str.startswith("tensor("):
            return (formal.type_str,)
        for constraint in schema.type_constraints:
            if constraint.type_param_str == formal.type_str:
                return tuple(constraint.allowed_type_strs)
        return ()

    @staticmethod
    def _schema_formal_at(formals, idx: int):
        if idx < len(formals):
            return formals[idx]
        if formals and formals[-1].option == onnx.defs.OpSchema.FormalParameterOption.Variadic:
            return formals[-1]
        return None

    @staticmethod
    def _tensor_onnx_dtype(tensor: gs.Variable | gs.Constant):
        dtype = getattr(tensor, "export_dtype", None) or tensor.dtype
        for onnx_dtype in _ONNX_TENSOR_TYPE_STRINGS:
            if _is_same_dtype(dtype, onnx_dtype):
                return onnx_dtype
        return None

    def _schema_dtype_pairs(self) -> tuple[tuple[onnx.TensorProto.DataType, onnx.TensorProto.DataType], ...]:
        return ((self._original_onnx_dtype, self._export_onnx_dtype),)

    def _requires_schema_enforcement(
        self,
        tensor: gs.Variable | gs.Constant,
        allowed_types: tuple[str, ...],
    ) -> bool:
        tensor_dtype = self._tensor_onnx_dtype(tensor)
        if tensor_dtype is None:
            return False
        return any(
            _ONNX_TENSOR_TYPE_STRINGS[original_dtype] in allowed_types
            and _ONNX_TENSOR_TYPE_STRINGS[export_dtype] not in allowed_types
            and tensor_dtype in (original_dtype, export_dtype)
            for original_dtype, export_dtype in self._schema_dtype_pairs()
        )

    def _get_schema_enforced_io(self, node: gs.Node) -> _EnforcedIoSpec | None:
        if self._opset_version is None or node.domain not in (None, "", "ai.onnx"):
            return None
        try:
            schema = onnx.defs.get_schema(node.op, self._opset_version)
        except onnx.defs.SchemaError:
            return None

        def enforced_indices(tensors, formals) -> tuple[int, ...]:
            return tuple(
                idx
                for idx, tensor in enumerate(tensors)
                if (formal := self._schema_formal_at(formals, idx)) is not None
                and self._requires_schema_enforcement(
                    tensor, self._schema_type_options(schema, formal)
                )
            )

        inputs = enforced_indices(node.inputs, schema.inputs)
        outputs = enforced_indices(node.outputs, schema.outputs)
        return (inputs, outputs) if inputs or outputs else None

    def _required_enforced_io_dtype(self) -> tuple[onnx.TensorProto.DataType, str]:
        return self._original_onnx_dtype, self._original_dtype_str

    def _cast_enforced_input_edges(self) -> bool:
        return self._enforce_io_casts

    @classmethod
    @abstractmethod
    def allowed_dtypes(cls) -> tuple[str, ...]: ...

    @abstractmethod
    def _convert_graph(
        self,
        graph: gs.Graph,
    ): ...

    def _handle_cast(self, cast_node: gs.Node, graph: gs.Graph):
        assert isinstance(cast_node, gs.Node) and cast_node.op == "Cast", \
            f"Expected Cast node in {self.__class__.__name__}._handle_cast()"
        cast_out: gs.Variable = cast_node.outputs[0]
        if cast_out.name in self._convert_exceptions:
            logger.debug("Skipping dtype conversion of explicitly marked cast output '%s'", cast_out.name)
            return
        if self._direct_cast:
            cast_out.dtype = self._export_onnx_dtype
            cast_node.attrs["to"] = self._export_onnx_dtype
            logger.debug("Update Cast op '%s' to directly cast to %s", cast_node.name, self._export_dtype_str)
        else:
            self._convert_exceptions[cast_out.name] = "Cast output and direct_cast=False"
            consumers: list[gs.Node] = list(cast_out.outputs)
            out_new: gs.Variable = graph.layer(
                name=cast_out.name + f"_cast_{self._original_dtype_str}_to_{self._export_dtype_str}",
                op="Cast",
                inputs=[cast_out],
                outputs=[gs.Variable(cast_out.name + f"_{self._export_dtype_str}", dtype=self._export_onnx_dtype, shape=cast_out.shape)],
                attrs={"to": self._export_onnx_dtype},
            )[0]
            for consumer in consumers:
                for i, inp in enumerate(consumer.inputs):
                    if inp is cast_out:
                        consumer.inputs[i] = out_new
                        logger.debug("Add %s -> %s Cast node to feed '%s'", self._original_dtype_str, self._export_dtype_str, consumer.name)

    def _convert_cast_constant_inputs(self, cast_node: gs.Node, graph: gs.Graph):
        # Cast inputs are skipped by the per-node I/O pass, so literal constants
        # feeding a Cast would otherwise stay in the original dtype.
        for i, inp in enumerate(cast_node.inputs):
            if isinstance(inp, gs.Constant):
                self._convert_tensor(inp, cast_node, graph, i, is_attr=False)

    def _constant_fits_export(self, values: np.ndarray, np_type: np.dtype) -> bool:
        # Whether literal values survive the cast to the export dtype without
        # overflow or wraparound.
        if values.size == 0:
            return True
        if np.issubdtype(np_type, np.integer):
            info = np.iinfo(np_type)
            return bool(values.min() >= info.min and values.max() <= info.max)
        return self._float_constant_fits_export(values, np_type)

    def _float_constant_fits_export(self, values: np.ndarray, np_type: np.dtype) -> bool:
        return True

    def _report_constant_conversion(
        self,
        name: str,
        values: np.ndarray,
        converted_values: np.ndarray,
    ) -> None:
        if not np.issubdtype(values.dtype, np.floating):
            return

        original = values.astype(np.float64)
        converted = converted_values.astype(np.float64)
        absolute_error = np.abs(original - converted)
        relative_error = np.divide(
            absolute_error,
            np.abs(original),
            out=np.zeros_like(absolute_error),
            where=original != 0,
        )
        logger.info(
            "Constant '%s': values=%d max_abs_error=%g mean_abs_error=%g rmse=%g max_rel_error=%g",
            name,
            values.size,
            absolute_error.max(initial=0),
            absolute_error.mean() if absolute_error.size else 0,
            np.sqrt(np.mean(np.square(absolute_error))) if absolute_error.size else 0,
            relative_error.max(initial=0),
        )

    def _convert_tensor(
        self,
        tensor: gs.Variable | gs.Constant,
        node: gs.Node,
        graph: gs.Graph,
        idx: int | str,
        is_attr: bool
    ):
        tensor_dtype = getattr(tensor, "export_dtype", None) or tensor.dtype
        if not self._is_original_dtype(tensor_dtype):
            logger.debug("Skipping non-%s tensor '%s'", self._original_dtype_str, tensor.name)
            return
        if tensor.name in self._convert_exceptions:
            logger.debug("Skipping dtype conversion of explicitly marked tensor '%s' (%s)",
                         tensor.name, self._convert_exceptions[tensor.name])
            return

        if isinstance(tensor, gs.Variable):
            # special handling not needed for runtime tensors
            tensor.dtype = self._export_onnx_dtype
            logger.debug("Set dtype of tensor '%s' to %s", tensor.name, self._export_dtype_str)
        elif isinstance(tensor, gs.Constant):
            new_const_name: str = tensor.name + f"_{self._export_dtype_str}"
            if not (new_const := graph.tensors().get(new_const_name)):
                try:
                    np_type = onnx.helper.tensor_dtype_to_np_dtype(self._export_onnx_dtype)
                except (TypeError, ValueError, KeyError):
                    raise RuntimeError(f"Unsupported tensor datatype {self._export_dtype_str}")
                if not self._constant_fits_export(tensor.values, np_type):
                    raise ValueError(
                        f"Cannot convert constant '{tensor.name}' to {self._export_dtype_str}: "
                        "values exceed max_float or the export dtype range"
                    )
                cast_fn = getattr(self, "_cast_constant_values", None)
                cast_values = (
                    cast_fn(tensor.values, np_type)
                    if cast_fn is not None
                    else tensor.values.astype(np_type)
                )
                self._report_constant_conversion(tensor.name, tensor.values, cast_values)
                new_const = gs.Constant(new_const_name, cast_values)
            logger.debug("Add %s initializer '%s'", self._export_dtype_str, new_const.name)
            try:
                if is_attr:
                    assert isinstance(idx, str), "Node attribute index must be a string"
                    node.attrs[idx] = new_const
                    logger.debug("Set attr '%s' of node '%s' to '%s'", idx, node.name, new_const.name)
                else:
                    assert isinstance(idx, int), "Node input index must be an integer"
                    node.inputs[idx] = new_const
                    logger.debug("Set input %d of node '%s' to '%s'", idx, node.name, new_const.name)
            except (IndexError, ValueError, KeyError):
                typ = "attribute" if is_attr else "input"
                logger.exception("Failed to update %s %s ('%s') of node '%s'", typ, idx, tensor.name, node.name)
        else:
            logger.warning("Skipping conversion due to invalid tensor type '%s'", str(type(tensor)))

    def _convert_node_io(
        self,
        node: gs.Node,
        graph: gs.Graph,
        skip_names: set[str]
    ):
        for i, inp in enumerate(list(node.inputs)):
            if inp.name in skip_names:
                logger.debug("Skipping dtype conversion of graph input '%s'", inp.name)
                continue
            self._convert_tensor(inp, node, graph, i, is_attr=False)

        for out in node.outputs:
            assert isinstance(out, gs.Variable), f"Non gs.Variable output '{out.name}' ({type(out)}) for node '{node.name}'"
            if out.name in skip_names:
                logger.debug("Skipping dtype conversion of graph output '%s'", out.name)
                continue
            if out.name in self._convert_exceptions:
                logger.debug("Skipping dtype conversion of explicitly marked tensor '%s' (%s)",
                             out.name, self._convert_exceptions[out.name])
                continue
            if self._is_original_dtype(out.dtype):
                out.dtype = self._export_onnx_dtype
                if node.op == "Cast":
                    node.attrs["to"] = self._export_onnx_dtype

    def _enforced_io_indices(
        self,
        indices: _EnforcedIoIndices,
        tensors: Sequence[gs.Variable | gs.Constant],
    ) -> tuple[int, ...]:
        if self._ALL_IO_INDICES in indices:
            assert len(indices) == 1, "Invalid tensor indices: all marker cannot be combined with specific indices"
            return tuple(range(len(tensors)))
        return tuple(idx for idx in indices if 0 <= idx < len(tensors))

    def _is_enforced_input_index(self, node: gs.Node, idx: int) -> bool:
        if not (enforced := self._get_enforced_io(node)):
            return False
        return idx in self._enforced_io_indices(enforced[0], node.inputs)

    def _cast_constant_for_enforced_io(
        self,
        tensor: gs.Constant,
        node: gs.Node,
        idx: int,
        required_dtype: onnx.TensorProto.DataType,
        required_dtype_str: str,
    ) -> gs.Constant:
        try:
            np_type = onnx.helper.tensor_dtype_to_np_dtype(required_dtype)
        except (TypeError, ValueError, KeyError):
            raise RuntimeError(f"Unsupported tensor datatype {required_dtype_str}")
        new_const = gs.Constant(
            tensor.name + f"_{required_dtype_str}",
            tensor.values.astype(np_type),
        )
        node.inputs[idx] = new_const
        return new_const

    def _export_constant_for_enforced_input_cast(
        self,
        tensor: gs.Constant,
        graph: gs.Graph,
    ) -> gs.Constant | None:
        new_const_name = tensor.name + f"_{self._export_dtype_str}"
        if new_const := graph.tensors().get(new_const_name):
            return new_const
        try:
            np_type = onnx.helper.tensor_dtype_to_np_dtype(self._export_onnx_dtype)
        except (TypeError, ValueError, KeyError):
            raise RuntimeError(f"Unsupported tensor datatype {self._export_dtype_str}")
        if not self._constant_fits_export(tensor.values, np_type):
            self._convert_exceptions[tensor.name] = f"constant values outside {self._export_dtype_str} range"
            logger.warning(
                "Keeping enforced-input constant '%s' in %s: values exceed max_float "
                "or the %s range",
                tensor.name, self._original_dtype_str, self._export_dtype_str,
            )
            return None
        cast_fn = getattr(self, "_cast_constant_values", None)
        cast_values = (
            cast_fn(tensor.values, np_type)
            if cast_fn is not None
            else tensor.values.astype(np_type)
        )
        self._report_constant_conversion(tensor.name, tensor.values, cast_values)
        return gs.Constant(new_const_name, cast_values)

    def _mark_enforced_input(
        self,
        node: gs.Node,
        graph: gs.Graph,
        idx: int,
    ):
        tensor = node.inputs[idx]
        required_dtype, required_dtype_str = self._required_enforced_io_dtype()
        tensor_dtype = getattr(tensor, "export_dtype", None) or tensor.dtype
        if self._cast_enforced_input_edges() and isinstance(tensor, gs.Constant):
            cast_in = tensor
            if self._is_original_dtype(tensor_dtype):
                exported = self._export_constant_for_enforced_input_cast(tensor, graph)
                if exported is not None:
                    cast_in = exported
            cast_out: gs.Variable = graph.layer(
                name=f"{node.name or node.op}_input_{idx}_cast_{required_dtype_str}",
                op="Cast",
                inputs=[cast_in],
                outputs=[gs.Variable(
                    f"{tensor.name}_{node.name or node.op}_input_{idx}_{required_dtype_str}",
                    dtype=required_dtype,
                    shape=tensor.shape,
                )],
                attrs={"to": required_dtype},
            )[0]
            node.inputs[idx] = cast_out
            self._convert_exceptions[cast_out.name] = f"{node.op} input {idx} requires {required_dtype_str}"
            return
        if isinstance(tensor, gs.Constant):
            if not self._is_original_dtype(tensor_dtype):
                dtype_str = onnx.helper.tensor_dtype_to_string(tensor_dtype) \
                    if isinstance(tensor_dtype, int) else str(tensor_dtype)
                logger.warning(
                    "Promoting to %s: tensor '%s' (dtype: %s), required for input %d of %s op '%s'",
                    required_dtype_str, tensor.name, dtype_str, idx, node.op, node.name,
                )
                tensor = self._cast_constant_for_enforced_io(tensor, node, idx, required_dtype, required_dtype_str)
            self._convert_exceptions[tensor.name] = f"{node.op} input {idx} requires {required_dtype_str}"
            return

        if (
            self._cast_enforced_input_edges()
            and tensor.name in self._convert_exceptions
            and _is_same_dtype(tensor_dtype, required_dtype)
        ):
            return

        if not self._cast_enforced_input_edges():
            if not self._is_original_dtype(tensor_dtype):
                dtype_str = onnx.helper.tensor_dtype_to_string(tensor_dtype) \
                    if isinstance(tensor_dtype, int) else str(tensor_dtype)
                logger.warning(
                    "Promoting to %s: tensor '%s' (dtype: %s), required for input %d of %s op '%s'",
                    required_dtype_str, tensor.name, dtype_str, idx, node.op, node.name,
                )
                tensor.dtype = required_dtype
            self._convert_exceptions[tensor.name] = f"{node.op} input {idx} requires {required_dtype_str}"
            return

        if not self._convert_io and _is_same_dtype(tensor_dtype, required_dtype):
            # The edge already carries the required dtype (e.g. a model input
            # that keeps its public dtype): remember the requirement and let
            # _update_input keep this edge on the original-dtype tensor.
            self._convert_exceptions[tensor.name] = f"{node.op} input {idx} requires {required_dtype_str}"
            return

        cast_out: gs.Variable = graph.layer(
            name=f"{node.name or node.op}_input_{idx}_cast_{required_dtype_str}",
            op="Cast",
            inputs=[tensor],
            outputs=[gs.Variable(
                f"{tensor.name}_{node.name or node.op}_input_{idx}_{required_dtype_str}",
                dtype=required_dtype,
                shape=tensor.shape,
            )],
            attrs={"to": required_dtype},
        )[0]
        node.inputs[idx] = cast_out
        self._convert_exceptions[cast_out.name] = f"{node.op} input {idx} requires {required_dtype_str}"

    def _mark_enforced_output(
        self,
        node: gs.Node,
        graph: gs.Graph,
        idx: int,
    ):
        tensor = node.outputs[idx]
        required_dtype, required_dtype_str = self._required_enforced_io_dtype()
        tensor.dtype = required_dtype
        self._convert_exceptions[tensor.name] = f"{node.op} output {idx} requires {required_dtype_str}"

        if not self._enforce_io_casts:
            return

        consumer_edges = [
            (consumer, i)
            for consumer in list(tensor.outputs)
            for i, inp in enumerate(consumer.inputs)
            if inp is tensor and not self._is_enforced_input_index(consumer, i)
        ]
        if not consumer_edges:
            return
        cast_out: gs.Variable = graph.layer(
            name=f"{node.name or node.op}_output_{idx}_cast_{self._export_dtype_str}",
            op="Cast",
            inputs=[tensor],
            outputs=[gs.Variable(
                f"{tensor.name}_{self._export_dtype_str}",
                dtype=self._export_onnx_dtype,
                shape=tensor.shape,
            )],
            attrs={"to": self._export_onnx_dtype},
        )[0]
        for consumer, i in consumer_edges:
            consumer.inputs[i] = cast_out

    def _apply_enforced_io_exclusions(self, node: gs.Node, graph: gs.Graph):
        if not (enforced := self._get_enforced_io(node)):
            return
        assert len(enforced) >= 1, "Invalid enforced I/O tensor indices"
        inp_indices = enforced[0]
        out_indices = enforced[1] if len(enforced) > 1 else ()
        for idx in self._enforced_io_indices(inp_indices, node.inputs):
            self._mark_enforced_input(node, graph, idx)
        for idx in self._enforced_io_indices(out_indices, node.outputs):
            self._mark_enforced_output(node, graph, idx)

    def _update_input(self, graph: gs.Graph, inp: gs.Variable):
        if self._convert_io:
            if inp.name in self._convert_exceptions:
                logger.debug("Skipping dtype conversion of explicitly marked graph input '%s' (%s)",
                             inp.name, self._convert_exceptions[inp.name])
                return
            inp.dtype = self._export_onnx_dtype
            logger.debug("Set dtype to %s for input '%s'", self._export_dtype_str, inp.name)
            return
        # convert_io=False: keep the public input dtype. Edges that must stay
        # in the original dtype (enforced I/O marked during _convert_graph)
        # keep consuming the input directly; the other consumers get a cast.
        consumers = list(inp.outputs)
        if not consumers:
            return
        inp_new = graph.layer(
            name=inp.name + f"_to_{self._export_dtype_str}",
            op="Cast",
            inputs=[inp],
            outputs=[gs.Variable(inp.name + f"_{self._export_dtype_str}", dtype=self._export_onnx_dtype, shape=inp.shape)],
            attrs={"to": self._export_onnx_dtype}
        )[0]
        for node in consumers:
            for i, val in enumerate(node.inputs):
                if val is inp and not self._is_enforced_input_index(node, i):
                    node.inputs[i] = inp_new
                    logger.debug("Update node '%s' to accept %s input '%s'", node.name, self._export_dtype_str, inp_new.name)
        if inp.name not in self._convert_exceptions:
            self._convert_exceptions[inp.name] = "graph input and convert_io=False"

    def _update_output(self, graph: gs.Graph, out: gs.Variable, out_idx: int):
        if out.name in self._convert_exceptions:
            logger.debug("Skipping dtype conversion of explicitly marked graph output '%s' (%s)",
                         out.name, self._convert_exceptions[out.name])
            if self._convert_io:
                # keep the public output name on the converted tensor; the
                # enforced-dtype tensor the Cast consumes gets the suffix
                public_name = out.name
                internal_name = out.name + f"_{self._required_enforced_io_dtype()[1]}"
                self._convert_exceptions[internal_name] = self._convert_exceptions.pop(public_name)
                out.name = internal_name
                out_new = graph.layer(
                    name=public_name + f"_to_{self._export_dtype_str}",
                    op="Cast",
                    inputs=[out],
                    outputs=[gs.Variable(public_name, dtype=self._export_onnx_dtype, shape=out.shape)],
                    attrs={"to": self._export_onnx_dtype}
                )[0]
                graph.outputs[out_idx] = out_new
                logger.debug("Add %s cast node for output '%s'", self._export_dtype_str, public_name)
            return
        out.dtype = self._export_onnx_dtype
        logger.debug("Set dtype to %s for output '%s'", self._export_dtype_str, out.name)
        if not self._convert_io:
            # add cast nodes at model outputs
            out_name = out.name
            out.name = out.name + f"_{self._export_dtype_str}"
            out_new = graph.layer(
                name=out.name + f"_to_{self._original_dtype_str}",
                op="Cast",
                inputs=[out],
                outputs=[gs.Variable(out_name, dtype=self._original_onnx_dtype, shape=out.shape)],
                attrs={"to": self._original_onnx_dtype}
            )[0]
            graph.outputs[out_idx] = out_new
            logger.debug("Add %s cast node for output '%s'", self._original_dtype_str, out_name)
            self._convert_exceptions[out_name] = "graph output and convert_io=False"

    def _update_inputs(self, graph: gs.Graph):
        for graph_inp in list(graph.inputs):
            if not self._is_original_dtype(graph_inp.dtype):
                logger.debug("Skipping non-%s input '%s'", self._original_dtype_str, graph_inp.name)
                continue
            self._update_input(graph, graph_inp)

    def _update_outputs(self, graph: gs.Graph):
        for i, graph_out in enumerate(list(graph.outputs)):
            if not self._is_original_dtype(graph_out.dtype):
                logger.debug("Skipping non-%s output '%s'", self._original_dtype_str, graph_out.name)
                continue
            self._update_output(graph, graph_out, i)

    def _fold_literal_constants(self, graph: gs.Graph):
        for node in graph.nodes:
            if node.op != "Constant" or node.inputs:
                continue
            if not isinstance((value := node.attrs.get("value")), gs.Constant):
                continue
            const_node_out: gs.Variable = node.outputs[0]
            const_init: gs.Constant = gs.Constant(
                name=(const_node_out.name or node.name) + "_folded",
                values=value.values
            )
            consumers: list[gs.Node] = list(const_node_out.outputs)
            for consumer in consumers:
                for i, inp in enumerate(consumer.inputs):
                    if inp is const_node_out:
                        consumer.inputs[i] = const_init
            node.outputs.clear()
            logger.debug(
                "Graph '%s': folded literal Constant producer node for '%s' into graph initializer",
                graph.name, const_node_out.name
            )
        graph = graph.cleanup(
            remove_unused_graph_inputs=True,
            remove_unused_node_outputs=self._remove_unused_node_outputs,
        ).toposort()

    def _check_tensor_dtypes(self, graph: gs.Graph) -> tuple[list[str], list[str]]:
        all_tensors: list[str] = []
        not_converted: list[str] = []
        for tensor_name, tensor in graph.tensors().items():
            tensor_dtype = getattr(tensor, "export_dtype", None) or tensor.dtype
            if self._is_original_dtype(tensor_dtype):
                if (exc_reason := self._convert_exceptions.get(tensor_name)) is None:
                    logger.warning(
                        "Graph '%s': tensor '%s' not converted to %s (unhandled)",
                        graph.name, tensor_name, self._export_dtype_str
                    )
                else:
                    logger.debug(
                        "Graph '%s': tensor '%s' not converted to %s (%s)",
                        graph.name, tensor_name, self._export_dtype_str, exc_reason
                    )
                not_converted.append(tensor_name)
            all_tensors.append(tensor_name)
        return all_tensors, not_converted

    def _check_conversion(self, graph: gs.Graph) -> tuple[list[str], list[str]]:
        graph.cleanup(
            remove_unused_graph_inputs=True,
            remove_unused_node_outputs=self._remove_unused_node_outputs,
        ).toposort()
        return self._check_tensor_dtypes(graph)

    def convert_model(
        self,
        input_model: onnx.ModelProto
    ) -> onnx.ModelProto:
        self._opset_version = next(
            (
                opset.version
                for opset in input_model.opset_import
                if opset.domain in ("", "ai.onnx")
            ),
            None,
        )
        try:
            input_model = onnx.shape_inference.infer_shapes(
                input_model,
                check_type=True,
                strict_mode=True,
            )
        except onnx.shape_inference.InferenceError as e:
            # No traceback: the error text is in the message, malformed
            # models still fail at the final inference below, and this also
            # fires for expected transients (weight-only quantization holds
            # fp32 activations against bf16-dequantized weights before the
            # I/O conversion completes).
            logger.warning(
                "ONNX shape inference failed; proceeding with original model: %s",
                e,
            )
        try:
            onnx.checker.check_model(input_model, full_check=True)
        except (onnx.checker.ValidationError, Exception) as e:
            if "InferenceError" in type(e).__name__ or isinstance(
                e, onnx.checker.ValidationError
            ):
                logger.warning(
                    "ONNX model validation failed; proceeding with original model: %s",
                    e,
                )
            else:
                raise

        root, *subgraphs = self._collect_all_graphs(gs.import_onnx(input_model))
        all_tensors: list[str] = []
        not_converted: list[str] = []

        # convert subgraphs (if any)
        for g in subgraphs:
            self._fold_literal_constants(g)
            self._convert_graph(g)
            self._update_inputs(g)
            self._update_outputs(g)
            conv_info = self._check_conversion(g)
            all_tensors += conv_info[0]
            not_converted += conv_info[1]

        # convert root graph
        self._fold_literal_constants(root)
        self._convert_graph(root)
        self._update_inputs(root)
        self._update_outputs(root)
        conv_info = self._check_conversion(root)
        all_tensors += conv_info[0]
        not_converted += conv_info[1]

        new_model = onnx.shape_inference.infer_shapes(
            gs.export_onnx(root), check_type=True, strict_mode=True, data_prop=len(subgraphs) == 0
        )
        new_model.ir_version = input_model.ir_version
        onnx.checker.check_model(new_model, full_check=True)

        all_tensors = set(all_tensors)
        not_converted = set(not_converted)
        total = len(all_tensors)
        failed = len(not_converted)
        converted = total - failed

        if total == 0:
            logger.info(
                "Conversion results: graph has no %s tensors to convert to %s",
                self._original_dtype_str, self._export_dtype_str
            )
        else:
            pct = (converted / total) * 100.0
            logger.info(
                "Conversion results: converted %d/%d (%.2f%%) %s tensors to %s",
                converted, total, pct, self._original_dtype_str, self._export_dtype_str
            )

        return new_model


class FP32Converter(OnnxDtypeConverterBase):

    _ALLOWED_BF16_ROUNDING: Final[tuple[str, ...]] = ("nearest", "truncate")

    def __init__(
        self,
        export_dtype: str,
        convert_io: bool = False,
        direct_cast: bool = True,
        remove_unused_node_outputs: bool = False,
        bf16_rounding: str = "nearest",
        max_float: float = 1e9,
    ):
        if bf16_rounding not in self._ALLOWED_BF16_ROUNDING:
            raise ValueError(
                f"Invalid bf16_rounding '{bf16_rounding}', select from {list(self._ALLOWED_BF16_ROUNDING)}"
            )
        self._bf16_rounding = bf16_rounding
        self._max_float = max_float
        super().__init__(
            "fp32",
            export_dtype,
            convert_io,
            direct_cast,
            remove_unused_node_outputs,
            enforce_io_casts=True,
        )

    @classmethod
    def allowed_dtypes(cls) -> tuple[str, ...]:
        return ("bf16", "fp16")

    # numpy 2.x neither knows ml_dtypes.bfloat16 to finfo nor classifies it as
    # floating, so the float export maxima are listed explicitly.
    _FLOAT_EXPORT_MAX: Final[dict[str, float]] = {
        "bf16": 3.3895313892515355e38,
        "fp16": float(np.finfo(np.float16).max),
    }

    def _float_constant_fits_export(self, values: np.ndarray, np_type: np.dtype) -> bool:
        dtype_max = self._FLOAT_EXPORT_MAX.get(self._export_dtype_str)
        if dtype_max is None:
            return True
        limit = min(self._max_float, dtype_max)
        return bool(np.all(np.abs(values) <= limit))

    def _cast_constant_values(self, values: np.ndarray, np_type: np.dtype) -> np.ndarray:
        # Override fp32→bf16 cast when truncate-toward-zero is requested. Default astype
        # uses ml_dtypes.bfloat16 which rounds-to-nearest-even.
        if (
            self._export_onnx_dtype == onnx.TensorProto.BFLOAT16
            and self._bf16_rounding == "truncate"
            and values.dtype == np.float32
        ):
            u32 = np.ascontiguousarray(values).view(np.uint32)
            u16 = (u32 >> 16).astype(np.uint16)
            return u16.view(np_type).reshape(values.shape)
        return values.astype(np_type)

    def _make_default_constant_of_shape_value(self, node: gs.Node) -> gs.Constant:
        try:
            np_type = onnx.helper.tensor_dtype_to_np_dtype(self._export_onnx_dtype)
        except (TypeError, ValueError, KeyError):
            raise RuntimeError(f"Unsupported tensor datatype {self._export_dtype_str}")
        return gs.Constant(
            f"{node.name or node.op}_value_{self._export_dtype_str}",
            np.array([0], dtype=np_type),
        )

    def _convert_constant_of_shape_value(self, node: gs.Node, graph: gs.Graph):
        if (val := node.attrs.get("value")) is not None:
            self._convert_tensor(val, node, graph, "value", is_attr=True)
        elif self._is_original_dtype(node.outputs[0].dtype):
            node.attrs["value"] = self._make_default_constant_of_shape_value(node)

    def _convert_random_like_dtype_attr(self, node: gs.Node):
        if self._is_original_dtype(node.attrs.get("dtype")):
            node.attrs["dtype"] = self._export_onnx_dtype

    def _convert_layer_normalization_stash_type(self, node: gs.Node):
        if self._export_onnx_dtype != onnx.TensorProto.BFLOAT16:
            return
        if "stash_type" in node.attrs and not self._is_original_dtype(node.attrs["stash_type"]):
            return

        node.attrs["stash_type"] = onnx.TensorProto.BFLOAT16
        logger.info(
            "LayerNormalization op '%s': converting default/fp32 stash_type to bf16 for "
            "Torq bf16 compatibility. This changes ONNX stage-one normalization compute "
            "precision from fp32 to bf16; validate model accuracy on the target backend.",
            node.name,
        )

    def _convert_graph(
        self,
        graph: gs.Graph
    ):
        skip_names = {i.name for i in graph.inputs} | {o.name for o in graph.outputs}
        for node in list(graph.nodes):
            # special case: Cast -> properly handle casts to fp32
            if node.op == "Cast" and self._is_original_dtype(node.outputs[0].dtype):
                if node.outputs[0].name in skip_names and not self._convert_io:
                    # This Cast is the model's output boundary: keep the declared
                    # I/O dtype and let the Cast itself convert at the boundary.
                    self._convert_cast_constant_inputs(node, graph)
                    self._convert_exceptions[node.outputs[0].name] = "graph output and convert_io=False"
                    continue
                self._handle_cast(node, graph)
                self._convert_cast_constant_inputs(node, graph)
                continue

            # special case: Constant -> constant value stored as an attribute
            if node.op == "Constant" and (val := node.attrs.get("value")) is not None:
                self._convert_tensor(val, node, graph, "value", is_attr=True)

            # special case: ConstantOfShape -> output dtype is inferred from the value attribute
            if node.op == "ConstantOfShape":
                self._convert_constant_of_shape_value(node, graph)

            # special case: Random*Like -> output dtype is controlled by an optional dtype attribute
            if node.op in {"RandomUniformLike", "RandomNormalLike"}:
                self._convert_random_like_dtype_attr(node)

            # special case: LayerNormalization -> stash_type controls stage-one compute dtype
            if node.op == "LayerNormalization":
                self._convert_layer_normalization_stash_type(node)

            self._apply_enforced_io_exclusions(node, graph)

            # special case: Resize -> only input and output can be cast to bf16
            if node.op == "Resize":
                if node.inputs[0].name not in skip_names:
                    self._convert_tensor(node.inputs[0], node, graph, 0, is_attr=False)
                if self._is_original_dtype((out := node.outputs[0]).dtype) and out.name not in skip_names:
                    out.dtype = self._export_onnx_dtype
                continue

            self._convert_node_io(node, graph, skip_names)

        logger.info("Updated graph '%s' dtypes to %s", graph.name, self._export_dtype_str)


class Int64Converter(OnnxDtypeConverterBase):

    def __init__(
        self,
        export_dtype: str = "int32",
        convert_io: bool = False,
        direct_cast: bool = True,
        remove_unused_node_outputs: bool = False,
        enforce_io_casts: bool = False,
    ):
        super().__init__(
            "int64",
            export_dtype,
            convert_io,
            direct_cast,
            remove_unused_node_outputs,
            enforce_io_casts=enforce_io_casts,
        )

        self._original_sdtype_str  = self._original_dtype_str
        self._original_udtype_str  = "u" + self._original_dtype_str
        self._export_sdtype_str    = self._export_dtype_str
        self._export_udtype_str    = "u" + self._export_dtype_str

        self._original_onnx_sdtype = self._original_onnx_dtype
        self._original_onnx_udtype = self._validate_dtype(self._original_udtype_str)
        self._export_onnx_sdtype   = self._export_onnx_dtype
        self._export_onnx_udtype   = self._validate_dtype(self._export_udtype_str)

    @classmethod
    def allowed_dtypes(cls) -> tuple[str, ...]:
        return ("int32", "int16", "int8")

    def _required_enforced_io_dtype(self) -> tuple[onnx.TensorProto.DataType, str]:
        self._set_dtypes(False)
        return onnx.TensorProto.INT64, "int64"

    def _is_original_dtype(self, dtype) -> bool:
        return (
            _is_same_dtype(dtype, self._original_onnx_sdtype)
            or _is_same_dtype(dtype, self._original_onnx_udtype)
        )

    def _is_unsigned(self, dtype) -> bool:
        return _is_same_dtype(dtype, self._original_onnx_udtype) or _is_same_dtype(dtype, self._export_onnx_udtype)

    def _schema_dtype_pairs(self) -> tuple[tuple[onnx.TensorProto.DataType, onnx.TensorProto.DataType], ...]:
        return (
            (self._original_onnx_sdtype, self._export_onnx_sdtype),
            (self._original_onnx_udtype, self._export_onnx_udtype),
        )

    def _constant_fits_export(self, values: np.ndarray, np_type: np.dtype) -> bool:
        if super()._constant_fits_export(values, np_type):
            return True
        if (
            self._original_onnx_dtype != onnx.TensorProto.INT64
            or self._export_onnx_dtype != onnx.TensorProto.INT32
        ):
            return False
        return bool(np.all(
            ((values >= np.iinfo(np.int32).min) & (values <= np.iinfo(np.int32).max))
            | (values == np.iinfo(np.int64).min)
            | (values == np.iinfo(np.int64).max)
        ))

    def _cast_constant_values(self, values: np.ndarray, np_type: np.dtype) -> np.ndarray:
        if (
            self._original_onnx_dtype == onnx.TensorProto.INT64
            and self._export_onnx_dtype == onnx.TensorProto.INT32
        ):
            values = values.copy()
            sentinel_mask = (values == np.iinfo(np.int64).min) | (values == np.iinfo(np.int64).max)
            if sentinel_mask.any():
                values[values == np.iinfo(np.int64).min] = np.iinfo(np.int32).min
                values[values == np.iinfo(np.int64).max] = np.iinfo(np.int32).max
        return values.astype(np_type)

    def _report_constant_conversion(
        self,
        name: str,
        values: np.ndarray,
        converted_values: np.ndarray,
    ) -> None:
        super()._report_constant_conversion(name, values, converted_values)
        if (
            self._original_onnx_dtype == onnx.TensorProto.INT64
            and self._export_onnx_dtype == onnx.TensorProto.INT32
        ):
            sentinel_count = np.count_nonzero(
                (values == np.iinfo(np.int64).min) | (values == np.iinfo(np.int64).max)
            )
            if sentinel_count:
                logger.info(
                    "Constant '%s': clamped %d INT64_MIN/MAX sentinel value(s) to INT32_MIN/MAX",
                    name,
                    sentinel_count,
                )

    def _set_dtypes(self, unsigned: bool):
        if unsigned:
            self._original_dtype_str  = self._original_udtype_str
            self._export_dtype_str    = self._export_udtype_str
            self._original_onnx_dtype = self._original_onnx_udtype
            self._export_onnx_dtype   = self._export_onnx_udtype
        else:
            self._original_dtype_str  = self._original_sdtype_str
            self._export_dtype_str    = self._export_sdtype_str
            self._original_onnx_dtype = self._original_onnx_sdtype
            self._export_onnx_dtype   = self._export_onnx_sdtype

    def _handle_cast(self, cast_node: gs.Node, graph: gs.Graph):
        self._set_dtypes(self._is_unsigned(cast_node.outputs[0].dtype))
        return super()._handle_cast(cast_node, graph)

    def _convert_tensor(
        self,
        tensor: gs.Variable | gs.Constant,
        node: gs.Node,
        graph: gs.Graph,
        idx: int | str,
        is_attr: bool
    ):
        dtype = getattr(tensor, "export_dtype", None) or tensor.dtype
        self._set_dtypes(self._is_unsigned(dtype))
        return super()._convert_tensor(tensor, node, graph, idx, is_attr)

    def _mark_enforced_input(self, node: gs.Node, graph: gs.Graph, idx: int):
        tensor_dtype = getattr(node.inputs[idx], "export_dtype", None) or node.inputs[idx].dtype
        self._set_dtypes(self._is_unsigned(tensor_dtype))
        return super()._mark_enforced_input(node, graph, idx)

    def _mark_enforced_output(self, node: gs.Node, graph: gs.Graph, idx: int):
        self._set_dtypes(self._is_unsigned(node.outputs[idx].dtype))
        return super()._mark_enforced_output(node, graph, idx)

    def _convert_node_io(
        self,
        node: gs.Node,
        graph: gs.Graph,
        skip_names: set[str]
    ):
        for i, inp in enumerate(list(node.inputs)):
            if inp.name in skip_names:
                logger.debug("Skipping dtype conversion of model input '%s'", inp.name)
                continue
            self._convert_tensor(inp, node, graph, i, is_attr=False)

        for out in node.outputs:
            assert isinstance(out, gs.Variable), f"Non gs.Variable output '{out.name}' ({type(out)}) for node '{node.name}'"
            if out.name in skip_names:
                logger.debug("Skipping dtype conversion of model output '%s'", out.name)
                continue
            if out.name in self._convert_exceptions:
                logger.debug("Skipping dtype conversion of explicitly marked tensor '%s' (%s)",
                             out.name, self._convert_exceptions[out.name])
                continue
            if self._is_original_dtype(out.dtype):
                self._set_dtypes(self._is_unsigned(out.dtype))
                out.dtype = self._export_onnx_dtype
                if node.op == "Cast":
                    node.attrs["to"] = self._export_onnx_dtype

    def _convert_graph(
        self,
        graph: gs.Graph
    ):
        skip_names = {i.name for i in graph.inputs} | {o.name for o in graph.outputs}
        for node in list(graph.nodes):
            if node.op == "Cast" and self._is_original_dtype(node.outputs[0].dtype):
                if node.outputs[0].name in skip_names and not self._convert_io:
                    # This Cast is the model's output boundary: keep the declared
                    # I/O dtype and let the Cast itself convert at the boundary.
                    self._convert_cast_constant_inputs(node, graph)
                    self._convert_exceptions[node.outputs[0].name] = "graph output and convert_io=False"
                    continue
                self._handle_cast(node, graph)
                self._convert_cast_constant_inputs(node, graph)
                continue

            if node.op == "Constant" and not node.name and not node.inputs:
                if self._is_original_dtype(node.outputs[0].dtype):
                    logger.warning(
                        "Skipping int64 tensor %s originating from unnamed Constant node",
                        node.outputs[0].name
                    )
                    self._convert_exceptions[node.outputs[0].name] = \
                        "int64 output from unnamed Constant node"
                    continue

            self._apply_enforced_io_exclusions(node, graph)
            self._convert_node_io(node, graph, skip_names)

        logger.info("Updated graph '%s' integer dtypes to %s", graph.name, self._export_dtype_str)

    def _update_input(self, graph: gs.Graph, inp: gs.Variable):
        self._set_dtypes(self._is_unsigned(inp.dtype))
        return super()._update_input(graph, inp)

    def _update_output(self, graph: gs.Graph, out: gs.Variable, out_idx: int):
        self._set_dtypes(self._is_unsigned(out.dtype))
        return super()._update_output(graph, out, out_idx)

    def _check_tensor_dtypes(self, graph: gs.Graph) -> tuple[list[str], list[str]]:
        self._export_dtype_str = self._export_sdtype_str
        return super()._check_tensor_dtypes(graph)


def _convert_modelopt(
    input_model: str | os.PathLike,
    convert_dtype: str,
    convert_io: bool,
    max_float: float,
):
    allowed_modelopt_dtypes = {"fp16", "bf16"}
    if convert_dtype not in allowed_modelopt_dtypes:
        raise ValueError(
            f"Invalid convert dtype '{convert_dtype}' for modelopt, "
            f"choose from {sorted(allowed_modelopt_dtypes)}"
        )
    try:
        from modelopt.onnx import autocast
    except ImportError as exc:
        raise RuntimeError(
            "Requested use_modelopt=True but TensorRT Model Optimizer "
            "('modelopt') is not installed. Install it or set use_modelopt=False."
        ) from exc

    return autocast.convert_to_mixed_precision(
        str(input_model),
        convert_dtype,
        data_max=max_float,
        init_max=max_float,
        keep_io_types=not convert_io,
    )


def convert_onnx_model(
    input_model: onnx.ModelProto,
    convert_dtype: str,
    convert_io: bool,
    target_opset: int,
    remove_unused_node_outputs: bool,
    bf16_rounding: str = "nearest",
    enforce_io_casts: bool = False,
    max_float: float = 1e9,
):
    """Convert an in-memory ONNX model with the internal dtype converter."""
    model = _upgrade_model(input_model, target_opset)

    if convert_dtype in FP32Converter.allowed_dtypes():
        converter = FP32Converter(
            convert_dtype,
            convert_io=convert_io,
            remove_unused_node_outputs=remove_unused_node_outputs,
            bf16_rounding=bf16_rounding,
            max_float=max_float,
        )
    elif convert_dtype in Int64Converter.allowed_dtypes():
        converter = Int64Converter(
            convert_dtype,
            convert_io=convert_io,
            remove_unused_node_outputs=remove_unused_node_outputs,
            enforce_io_casts=enforce_io_casts,
        )
    else:
        allowed_dtypes = list(FP32Converter.allowed_dtypes()) + list(
            Int64Converter.allowed_dtypes()
        )
        raise ValueError(
            f"Invalid convert dtype '{convert_dtype}', choose from {allowed_dtypes}"
        )

    return converter.convert_model(model)


def convert_model(
    input_model: str | os.PathLike,
    output_model: str | os.PathLike,
    convert_dtype: str,
    use_modelopt: bool = False,
    convert_io: bool = False,
    max_float: float = 1e9,
    target_opset: int = 22,
    torq_onnx_finalize: bool = False,
    remove_unused_node_outputs: bool = False,
    bf16_rounding: str = "nearest",
    enforce_io_casts: bool = False,
):
    if use_modelopt:
        if bf16_rounding != "nearest":
            raise ValueError("bf16_rounding is only supported with the internal converter (--modelopt is incompatible)")
        converted_model = _convert_modelopt(
            input_model,
            convert_dtype=convert_dtype,
            convert_io=convert_io,
            max_float=max_float,
        )
    else:
        converted_model = convert_onnx_model(
            onnx.load(input_model),
            convert_dtype=convert_dtype,
            convert_io=convert_io,
            target_opset=target_opset,
            remove_unused_node_outputs=remove_unused_node_outputs,
            bf16_rounding=bf16_rounding,
            enforce_io_casts=enforce_io_casts,
            max_float=max_float,
        )

    export_dir = Path(output_model).parent
    export_dir.mkdir(parents=True, exist_ok=True)
    if torq_onnx_finalize:
        converted_model = _finalize_torq_ready_onnx(converted_model)
    onnx.save(converted_model, output_model)
    logger.info("Saved converted model to '%s'", str(output_model))


def add_onnx_dtype_convert_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "-i", "--input",
        type=str,
        required=True,
        help="Input ONNX model path"
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Output ONNX model path"
    )
    parser.add_argument(
        "-d", "--dtype",
        type=str,
        metavar="DTYPE",
        choices=FP32Converter.allowed_dtypes() + Int64Converter.allowed_dtypes(),
        required=True,
        help="Export data type (choices: %(choices)s)"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=22,
        help="ONNX opset to use; note that a relatively new opset is required for bf16 support in some ops (default: %(default)s)"
    )
    parser.add_argument(
        "--max-float",
        type=float,
        default=1e9,
        help=(
            "Maximum FP32 magnitude for constant conversion: initializers with values "
            "above `max_float` or the export dtype's range are rejected to avoid an "
            "overflowing conversion. With --modelopt it is also passed as data_max/init_max"
        )
    )
    parser.add_argument(
        "--convert-io",
        action="store_true",
        default=False,
        help="Convert model I/O to export dtype"
    )
    parser.add_argument(
        "--modelopt",
        action="store_true",
        default=False,
        help="Use TensorRT modelopt for dtype conversion"
    )
    parser.add_argument(
        "--bf16-rounding",
        type=str,
        choices=FP32Converter._ALLOWED_BF16_ROUNDING,
        default="nearest",
        help="Rounding mode for fp32→bf16 constants (default: %(default)s). 'truncate' drops the low 16 mantissa bits.",
    )
    parser.add_argument(
        "--enforce-io-casts",
        action="store_true",
        default=False,
        help=(
            "Insert Cast nodes so ONNX spec-mandated int64 operator I/O (e.g. Reshape "
            "shape, Slice params, Shape/Size outputs) stays int64. By default these "
            "tensors are preserved as int64 in place without adding casts."
        ),
    )
    parser.add_argument(
        "--strip-unused-outputs",
        action="store_true",
        default=False,
        help=(
            "Strip unused node outputs during post-conversion cleanup. "
            "WARNING: may cause errors with fixed output nodes like TopK. "
        )
    )
    parser.add_argument(
        "--torq-onnx-finalize",
        action="store_true",
        default=False,
        help=(
            "Run Torq-oriented ONNX post-process (ORT symbolic shapes, IR cap, "
            "value_info cleanup)"
        ),
    )
    from torq.lab.logging import add_logging_args
    add_logging_args(parser)


def onnx_dtype_convert_from_args(args: argparse.Namespace):
    from torq.lab.logging import configure_logging
    configure_logging(args.logging)
    convert_model(
        args.input,
        args.output,
        args.dtype,
        args.modelopt,
        args.convert_io,
        args.max_float,
        args.opset,
        torq_onnx_finalize=args.torq_onnx_finalize,
        remove_unused_node_outputs=args.strip_unused_outputs,
        bf16_rounding=args.bf16_rounding,
        enforce_io_casts=args.enforce_io_casts,
    )


def main(argv: list[str] | None = None) -> int:
    """Main entry point for torq-convert-dtype."""
    parser = argparse.ArgumentParser(
        prog="torq-convert-dtype",
        description="Convert models to Torq compatible dtypes",
    )
    model_type = parser.add_subparsers(dest="model_type", required=True)

    onnx_type = model_type.add_parser("onnx", help="Convert ONNX models")
    add_onnx_dtype_convert_args(onnx_type)

    args = parser.parse_args(argv)

    if args.model_type == "onnx":
        onnx_dtype_convert_from_args(args)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
