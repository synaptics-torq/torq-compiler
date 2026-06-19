import copy as _copy
import json
from dataclasses import dataclass
from functools import wraps
import subprocess
import sys
from pathlib import Path
from typing import Any
import numpy as np
import onnx
import onnxruntime
import pytest
from onnx import helper, numpy_helper, shape_inference, TensorProto, AttributeProto

from torq.testing.cases import Case
from torq.testing.hf import get_hf_model_file

from .versioned_fixtures import (
    versioned_cached_data_fixture,
    versioned_generated_file_fixture,
    versioned_hashable_object_fixture,
    versioned_static_file_fixture,
    versioned_unhashable_object_fixture,
    VersionedUncachedData,
)

"""
Fixtures and utilities for testing ONNX models.
"""
#
# File structure ----------------------------------------------------------
#   Classes              ModelWithMetadata, OnnxLayerCase
#   pytest hooks         pytest_addoption
#   Node / name utils    _source_node_name, _format_layer_node_name
#   Model signature      _vi_dtype_shape .. model_signature
#   Shape helpers        _has_dynamic_dims, _get_static_shape, _set_static_dims
#   ONNX attr helpers    _get_attr_i, _get_attr_ints, _get_attr_tensor
#   Value-info helper    _make_value_info_from_initializer
#   Tensor folding       _CAST_DTYPE .. _make_tensor_folder
#   Core extraction      _build_model_from_node_subset
#   Layer extraction     generate_onnx_layers_from_model, extract_onnx_subgraph
#   Model loading        get_full_model
#   BF16 conversion      _float32_to_bfloat16 .. convert_fp32_to_bf16
#   Pytest fixtures      onnx_model, onnx_model_file, ...
#

_pytest_config = None

# ---- Classes ----

class ModelWithMetadata:
    """Wrapper for ONNX model with metadata like node_index.

    ONNX protobuf objects don't allow setting arbitrary attributes,
    so we use this wrapper to store additional metadata.
    """

    def __init__(self, model: Any = None, node_index: int = None, path=None):
        self._model = model
        self.node_index = node_index
        self.path = str(path) if path is not None else None

    def __getattr__(self, name):
        # Delegate attribute access to wrapped model
        return getattr(self.model, name)

    def __getitem__(self, key):
        # Support dict-style access (e.g. layer["model"]) for backward compatibility
        return getattr(self, key)

    @property
    def model(self):
        if self._model is None and self.path is not None:
            self._model = onnx.load(self.path)
        return self._model


@dataclass
class OnnxLayerCase(Case):
    """Represents a generated ONNX layer test case plus source node name."""

    node_name: str = ""
    is_full_model: bool = False


# ---- pytest hooks ----

def pytest_addoption(parser):
    parser.addoption(
        "--onnx-print-layer-info",
        action="store_true",
        default=False,
        help="Print original ONNX node names for generated ONNX layer tests during setup",
    )


# ---- Node / name utilities ----

def _source_node_name(node, node_index):
    if node.name:
        return node.name

    outputs = [output_name for output_name in node.output if output_name]
    if outputs:
        return outputs[0]

    return f"{node.op_type}@{node_index}"


def _format_layer_node_name(case: OnnxLayerCase) -> str:
    if not case.node_name:
        return "ONNX node: <unknown>"

    label = "nodes" if " -> " in case.node_name else "node"
    return f"ONNX {label}: {case.node_name}"


# ---- Model signature ----

def _vi_dtype_shape(vi):
    try:
        tt = vi.type.tensor_type
    except Exception:
        return None, None

    elem = getattr(tt, 'elem_type', None)

    shape = None
    try:
        dims = []
        for d in getattr(tt, 'shape').dim:
            val = getattr(d, 'dim_value', None)
            if val is None:
                val = getattr(d, 'dim_param', None)
            dims.append(val)
        shape = tuple(dims)
    except Exception:
        shape = None

    return int(elem) if elem is not None else None, shape


def _inputs_outputs_signature(graph):
    inputs = []
    for vi in graph.input:
        inputs.append(_vi_dtype_shape(vi))

    outputs = []
    for vi in graph.output:
        outputs.append(_vi_dtype_shape(vi))

    return inputs, outputs


def _init_signature(init):
    try:
        arr = numpy_helper.to_array(init)
        dtype = str(arr.dtype)
        shape = tuple(int(d) for d in arr.shape)
        return [dtype, shape]
    except Exception:
        return [init.data_type, tuple(getattr(init, 'dims', ()))]

def model_signature(model):
    graph = getattr(model, 'graph', model)
    inputs, outputs = _inputs_outputs_signature(graph)
    inits = [_init_signature(init) for init in graph.initializer]

    # signature_list = [inputs, outputs, len(node), node_op_type_list, initializer]
    return [inputs, outputs, len(list(graph.node)), [n.op_type for n in graph.node], inits]



# ---- Shape helpers -- used by extraction and propagation ----

def _has_dynamic_dims(tt):
    """Return True if the tensor_type has any dimension that is dynamic."""
    if not tt.shape.dim:
        return False
    return any(
        (d.dim_value == 0 and not d.dim_param) or d.dim_param
        for d in tt.shape.dim
    )


def _get_static_shape(tt):
    """Return list of dim_values if all dims are static, otherwise None."""
    if not tt.shape.dim:
        return None
    dims = []
    for d in tt.shape.dim:
        if d.dim_value > 0 or (d.dim_value == 0 and not d.dim_param):
            dims.append(d.dim_value)
        else:
            return None
    return dims


def _should_propagate_layer_shape(vi, shape):
    """Return True if a layer-extraction output shape may update full-model value_info."""
    if not shape:
        return False
    if vi is None:
        return True
    existing_dims = vi.type.tensor_type.shape.dim
    if not existing_dims:
        # Full-model shape inference left this empty; a single-op extraction
        # often infers the wrong rank (e.g. [560] instead of [1, 560, H]).
        return False
    if len(shape) != len(existing_dims):
        return False
    return True


def _set_static_dims(vi, shape):
    """Copy a list of static dims into a value_info.

    Respects the protobuf oneof: setting dim_value on each dim
    automatically clears any dim_param, avoiding the foot-gun where
    setting dim_param (even to '') silently clears dim_value.
    """
    tt = vi.type.tensor_type
    if not tt.shape.dim:
        del tt.shape.dim[:]
        for s in shape:
            tt.shape.dim.add().dim_value = s
    else:
        for i, s in enumerate(shape):
            if i < len(tt.shape.dim):
                tt.shape.dim[i].dim_value = s


# ---- ONNX attribute helpers ----

def _get_attr_i(node, name, default=None):
    """Extract an int attribute from an ONNX node, or *default*."""
    for attr in node.attribute:
        if attr.name == name:
            return attr.i
    return default


def _get_attr_ints(node, name):
    """Extract an 'ints' attribute as a Python list, or None."""
    for attr in node.attribute:
        if attr.name == name:
            return list(attr.ints)
    return None


def _get_attr_tensor(node, name):
    """Extract a tensor attribute as a numpy array, or None."""
    for attr in node.attribute:
        if attr.name == name and attr.type == AttributeProto.TENSOR:
            return numpy_helper.to_array(attr.t)
    return None


# ---- Value-info helper (used by extraction) ----

def _make_value_info_from_initializer(init):
    """Synthesize a value_info for an initializer with its concrete shape."""
    try:
        arr = numpy_helper.to_array(init)
        return helper.make_tensor_value_info(init.name, init.data_type, list(arr.shape))
    except Exception:
        return helper.make_tensor_value_info(init.name, TensorProto.FLOAT, [])


# ---- Tensor-folding (evaluates shape-producing ONNX subgraphs) ----

_CAST_DTYPE = {
    TensorProto.INT64:    np.int64,
    TensorProto.INT32:    np.int32,
    TensorProto.FLOAT:    np.float32,
    TensorProto.BFLOAT16: np.float16,
}


def _get_axes(node, _fold):
    """Resolve Unsqueeze/Squeeze axes from input tensor or attribute."""
    axes = _fold(node.input[1]) if len(node.input) > 1 else None
    if axes is not None:
        return [int(a) for a in np.atleast_1d(axes)]
    return _get_attr_ints(node, 'axes')


def _fold_binary(node, _fold, op):
    """Fold Mul or Equal: recursively evaluate both inputs, apply elementwise op."""
    a = _fold(node.input[0])
    b = _fold(node.input[1])
    if a is None or b is None:
        return None
    return op(a, b)


# Per-op folding handlers
# Each takes (producer_node, fold_fn, cache) -> np.ndarray | None.
# *fold_fn* is a callable that recursively folds a tensor name; it is
# produced by _make_tensor_folder() below and carries the extraction
# context (constant_tensors, initializers, full graph nodes).

def _fold_Shape(node, _fold, lookup_vi):
    """Fold Shape op: resolve dims from value_info."""
    vi = lookup_vi(node.input[0])
    if vi is None:
        return None
    dims = _get_static_shape(vi.type.tensor_type)
    return np.array(dims, dtype=np.int64) if dims is not None else None


def _fold_Slice(node, _fold):
    """Apply numpy slicing with negative-index handling."""
    data = _fold(node.input[0])
    starts = _fold(node.input[1])
    ends = _fold(node.input[2])
    if data is None or starts is None or ends is None:
        return None
    axes  = _fold(node.input[3]) if len(node.input) > 3 and node.input[3] else None
    steps = _fold(node.input[4]) if len(node.input) > 4 and node.input[4] else None

    starts, ends = np.atleast_1d(starts, ends)
    axes  = np.atleast_1d(axes)  if axes  is not None else np.arange(len(starts))
    steps = np.atleast_1d(steps) if steps is not None else np.ones(len(starts), dtype=int)

    slices = [slice(None)] * data.ndim
    for i in range(len(starts)):
        ax = int(axes[i])
        st, en, sp = int(starts[i]), int(ends[i]), int(steps[i])
        ax += data.ndim if ax < 0 else 0
        st += data.shape[ax] if st < 0 else 0
        en += data.shape[ax] if en < 0 else 0
        slices[ax] = slice(st, en, sp)
    return data[tuple(slices)]


def _fold_Concat(node, _fold):
    """np.concatenate all inputs along axis attr."""
    inputs = [_fold(inp) for inp in node.input]
    return np.concatenate(inputs, axis=_get_attr_i(node, 'axis', 0)) if all(x is not None for x in inputs) else None


def _fold_Gather(node, _fold):
    """np.take along axis attr."""
    data, indices = _fold(node.input[0]), _fold(node.input[1])
    return np.take(data, indices, axis=_get_attr_i(node, 'axis', 0)) if data is not None and indices is not None else None


def _fold_Unsqueeze(node, _fold):
    """Insert size-1 dims (sorted reverse to avoid index drift)."""
    data, axes = _fold(node.input[0]), _get_axes(node, _fold)
    if data is None or axes is None:
        return None
    for ax in sorted(axes, reverse=True):
        data = np.expand_dims(data, axis=ax + data.ndim + 1 if ax < 0 else ax)
    return data


def _fold_Squeeze(node, _fold):
    """Remove size-1 dims (sorted reverse); no axes = squeeze all."""
    data, axes = _fold(node.input[0]), _get_axes(node, _fold)
    if data is None:
        return None
    if axes is None:
        return np.squeeze(data)
    for ax in sorted(axes, reverse=True):
        data = np.squeeze(data, axis=ax + data.ndim if ax < 0 else ax)
    return data


def _fold_Cast(node, _fold):
    """Cast folded data to the dtype from to-attribute (`_CAST_DTYPE` map)."""
    data = _fold(node.input[0])
    if data is None:
        return None
    to_type = _get_attr_i(node, 'to')
    dtype = _CAST_DTYPE.get(to_type)
    return data.astype(dtype) if dtype else data


def _fold_ConstantOfShape(node, _fold):
    """Fill a shape (folded from input[0]) with the value-attribute scalar."""
    shape = _fold(node.input[0])
    if shape is None:
        return None
    shape = np.atleast_1d(shape).astype(np.int64)
    val = _get_attr_tensor(node, 'value')
    fill_val   = val.flatten()[0] if val is not None else 0
    fill_dtype = val.dtype           if val is not None else np.float32
    return np.full(shape, fill_val, dtype=fill_dtype)


def _fold_Reshape(node, _fold):
    """np.reshape to shape from input[1]."""
    data, shape = _fold(node.input[0]), _fold(node.input[1])
    return np.reshape(data, np.atleast_1d(shape).astype(np.int64)) if data is not None and shape is not None else None


def _fold_Transpose(node, _fold):
    """np.transpose with optional perm attr; default = reverse all dims."""
    data, perm = _fold(node.input[0]), _get_attr_ints(node, 'perm')
    return np.transpose(data, axes=perm if perm else None) if data is not None else None


def _fold_Where(node, _fold):
    """np.where(cond, x, y) with all three folded recursively."""
    cond, x, y = _fold(node.input[0]), _fold(node.input[1]), _fold(node.input[2])
    return np.where(cond, x, y) if cond is not None and x is not None and y is not None else None


# Base dispatch table -- everything except Shape (which needs a local
# value_info lookup that varies per extraction).
# Set of op types that can be constant-folded
FOLDABLE_OP_TYPES = frozenset([
    'Constant', 'Shape', 'Slice', 'Concat', 'Gather', 'Unsqueeze',
    'Squeeze', 'Cast', 'ConstantOfShape', 'Reshape', 'Transpose',
    'Mul', 'Equal', 'Where', 'Add', 'Sub', 'Div',
])

_FOLD_BASE = {
    'Slice':           _fold_Slice,
    'Concat':          _fold_Concat,
    'Gather':          _fold_Gather,
    'Unsqueeze':       _fold_Unsqueeze,
    'Squeeze':         _fold_Squeeze,
    'Cast':            _fold_Cast,
    'ConstantOfShape': _fold_ConstantOfShape,
    'Reshape':         _fold_Reshape,
    'Transpose':       _fold_Transpose,
    'Mul':             lambda n, f: _fold_binary(n, f, np.multiply),
    'Equal':           lambda n, f: _fold_binary(n, f, np.equal),
    'Where':           _fold_Where,
}


def _make_tensor_folder(*, constant_tensors, orig_initializers, all_nodes, fold_table):
    """Return a fold_fn(name=None) that evaluates shape subgraphs.

    The returned closure captures the extraction context so that the
    per-op handlers (which are pure module-level functions) stay
    independent of any particular model instance.
    """
    def _try_fold(name, _cache=None):
        """Fold tensor *name* recursively; caches results in _cache."""
        if _cache is None:
            _cache = {}
        if name in _cache:
            return _cache[name]
        if name in constant_tensors:
            return numpy_helper.to_array(constant_tensors[name])
        if name in orig_initializers:
            return numpy_helper.to_array(orig_initializers[name])

        producer = None
        for n in all_nodes:
            if name in n.output:
                producer = n
                break
        if producer is None:
            return None

        handler = fold_table.get(producer.op_type)
        if handler is None:
            return None

        try:
            result = handler(producer, _try_fold)
        except (IndexError, ValueError, TypeError):
            return None
        if result is not None:
            _cache[name] = result
        return result

    return _try_fold


# ---- Shape propagation for external inputs ----

def _resolve_tensor_shape(name, all_nodes, orig_value_info, orig_inputs,
                          orig_outputs, orig_initializers, constant_tensors,
                          _cache=None):
    """Compute the shape of a tensor by tracing back through producers.

    With ONNX shape inference run on the full model first, most tensors
    already have static shapes in value_info.  This function is a
    fallback for the few ops where ONNX leaves dynamic dims (unk__*):
    Reshape, Transpose, and Expand.

    Returns None if the shape cannot be resolved.
    """
    if _cache is None:
        _cache = {}
    if name in _cache:
        return _cache[name]

    # 1. Look up in value_info / inputs / outputs
    for d in [orig_value_info, orig_inputs, orig_outputs]:
        if name in d:
            vi = d[name]
            tt = vi.type.tensor_type
            if tt.shape.dim:
                dims = []
                all_static = True
                for dim in tt.shape.dim:
                    if dim.dim_value > 0:
                        dims.append(dim.dim_value)
                    else:
                        all_static = False
                        break
                if all_static:
                    _cache[name] = dims
                    return dims

    # 2. Look up in initializers
    if name in orig_initializers:
        dims = list(orig_initializers[name].dims)
        _cache[name] = dims
        return dims

    # 3. Look up in folded constants
    if name in constant_tensors:
        t = constant_tensors[name]
        dims = list(t.dims) if t.dims else list(numpy_helper.to_array(t).shape)
        _cache[name] = dims
        return dims

    # 4. Trace back through producer (only for ops ONNX can't fully infer)
    producer = None
    for n in all_nodes:
        if name in n.output:
            producer = n
            break
    if producer is None:
        return None

    op = producer.op_type
    result = None

    # Build a fold table that uses value_info for Shape (populated by
    # onnx.shape_inference.infer_shapes run on the full model).
    lookup_vi = lambda n: (orig_value_info.get(n) or
                           orig_inputs.get(n) or
                           orig_outputs.get(n))
    fold_table = {**_FOLD_BASE,
                  'Shape': lambda n, f: _fold_Shape(n, f, lookup_vi)}

    if op == 'Reshape':
        data_shape = _resolve_tensor_shape(
            producer.input[0], all_nodes, orig_value_info, orig_inputs,
            orig_outputs, orig_initializers, constant_tensors, _cache)
        if data_shape is None:
            return None
        _fold = _make_tensor_folder(
            constant_tensors=constant_tensors,
            orig_initializers=orig_initializers,
            all_nodes=all_nodes,
            fold_table=fold_table,
        )
        shape_val = _fold(producer.input[1])
        if shape_val is None:
            return None
        shape_arr = np.atleast_1d(shape_val).astype(np.int64).tolist()
        # Resolve -1
        if -1 in shape_arr:
            known_size = 1
            for s in shape_arr:
                if s != -1:
                    known_size *= s
            total_size = 1
            for s in data_shape:
                total_size *= s
            if total_size % known_size != 0:
                return None
            unknown_size = total_size // known_size
            shape_arr = [unknown_size if s == -1 else s for s in shape_arr]
        result = shape_arr

    elif op == 'Transpose':
        data_shape = _resolve_tensor_shape(
            producer.input[0], all_nodes, orig_value_info, orig_inputs,
            orig_outputs, orig_initializers, constant_tensors, _cache)
        if data_shape is None:
            return None
        perm = _get_attr_ints(producer, 'perm')
        if perm:
            result = [data_shape[i] for i in perm]
        else:
            result = data_shape[::-1]

    elif op == 'Expand':
        data_shape = _resolve_tensor_shape(
            producer.input[0], all_nodes, orig_value_info, orig_inputs,
            orig_outputs, orig_initializers, constant_tensors, _cache)
        _fold = _make_tensor_folder(
            constant_tensors=constant_tensors,
            orig_initializers=orig_initializers,
            all_nodes=all_nodes,
            fold_table=fold_table,
        )
        shape_val = _fold(producer.input[1])
        if data_shape is None or shape_val is None:
            return None
        target_shape = np.atleast_1d(shape_val).astype(np.int64).tolist()
        out = []
        max_ndim = max(len(data_shape), len(target_shape))
        for i in range(1, max_ndim + 1):
            d1 = data_shape[-i] if i <= len(data_shape) else 1
            t1 = target_shape[-i] if i <= len(target_shape) else 1
            if d1 == 1:
                out.append(t1)
            elif t1 == 1:
                out.append(d1)
            elif d1 == t1:
                out.append(d1)
            else:
                return None
        result = out[::-1]

    if result is not None:
        _cache[name] = result
    return result


# ---- Core extraction ----

def _build_model_from_node_subset(nodes_subset, all_nodes, model, graph_name,
                                   init_name_fn, log_prefix=""):
    """Build a standalone ONNX model from a subset of nodes.

    Shared logic between layer extraction and subgraph extraction.

    Args:
        nodes_subset: List of nodes to include in the new model.
        all_nodes: Full list of nodes from the original model (for determining
                   which outputs are consumed outside the subset).
        model: Original ONNX model (ModelProto or ModelWithMetadata).
        graph_name: Name for the new graph.
        init_name_fn: Callable(base_name, idx_init) -> str for renaming
                      initializers to avoid collisions.
        log_prefix: Prefix string for error messages.

    Returns:
        The inferred/checked ONNX model, or a deep copy of the raw model if
        inference fails.
    """
    # Unwrap ModelWithMetadata if needed
    if hasattr(model, 'model'):
        model = model.model

    graph = model.graph

    # Determine tensors produced and used by the subset
    produced = set()
    used = set()
    for n in nodes_subset:
        for o in n.output:
            produced.add(o)
        for inp in n.input:
            if inp:
                used.add(inp)

    external_inputs = used - produced

    # External outputs: produced names consumed by nodes outside the subset
    # or that are original model outputs
    external_outputs = set()
    orig_output_names = {o.name for o in graph.output}
    for name in produced:
        consumed_outside = False
        for other in all_nodes:
            if other in nodes_subset:
                continue
            if name in other.input:
                consumed_outside = True
                break
        if consumed_outside or (name in orig_output_names):
            external_outputs.add(name)

    # Build metadata maps
    orig_inputs = {vi.name: vi for vi in graph.input}
    orig_value_info = {vi.name: vi for vi in graph.value_info}
    orig_outputs = {vi.name: vi for vi in graph.output}
    orig_initializers = {init.name: init for init in graph.initializer}

    # Build map of Constant node outputs -> tensor values so we can preserve
    # them as initializers in extracted layers. This prevents dynamic shapes
    # (e.g. ConstantOfShape's shape input becoming an unknown external input).
    constant_tensors = {}
    for n in all_nodes:
        if n.op_type == 'Constant' and n.output:
            out_name = n.output[0]
            for attr in n.attribute:
                if attr.name == 'value' and attr.type == onnx.AttributeProto.TENSOR:
                    # Constant node TensorProto attributes often have empty name
                    # fields; copy the output name so renaming works later.
                    t = _copy.deepcopy(attr.t)
                    t.name = out_name
                    constant_tensors[out_name] = t
                elif attr.name == 'value_ints':
                    arr = np.array(attr.ints, dtype=np.int64)
                    constant_tensors[out_name] = numpy_helper.from_array(arr, name=out_name)
                elif attr.name == 'value_floats':
                    arr = np.array(attr.floats, dtype=np.float32)
                    constant_tensors[out_name] = numpy_helper.from_array(arr, name=out_name)
                elif attr.name == 'value_int':
                    arr = np.array([attr.i], dtype=np.int64)
                    constant_tensors[out_name] = numpy_helper.from_array(arr, name=out_name)
                elif attr.name == 'value_float':
                    arr = np.array([attr.f], dtype=np.float32)
                    constant_tensors[out_name] = numpy_helper.from_array(arr, name=out_name)


    # Tensor folding: evaluate shape-producing subgraphs
    # Build the per-extraction dispatch table (Shape needs a local
    # value_info lookup; the rest come from the module-level _FOLD_BASE).

    _lookup_vi = lambda name: (orig_value_info.get(name) or
                                orig_inputs.get(name) or
                                orig_outputs.get(name))
    _FOLD = {**_FOLD_BASE, 'Shape': lambda n, f: _fold_Shape(n, f, _lookup_vi)}

    _try_fold_tensor = _make_tensor_folder(
        constant_tensors=constant_tensors,
        orig_initializers=orig_initializers,
        all_nodes=all_nodes,
        fold_table=_FOLD,
    )

    for name in list(external_inputs):
        if name in constant_tensors or name in orig_initializers:
            continue
        folded = _try_fold_tensor(name)
        if folded is not None:
            constant_tensors[name] = numpy_helper.from_array(folded, name=name)
            if log_prefix:
                print(f'{log_prefix}: folded shape subgraph for {name} -> {folded.tolist()}')


    # Build new inputs/outputs/initializers/value_info
    new_inputs = []
    new_outputs = []
    new_initializers = []
    added_initializer_names = set()

    for name in sorted(external_inputs):
        if name in constant_tensors:
            # Preserve Constant node values and folded shape-subgraph outputs
            # as initializers so the compiler can inline them.  They are not
            # graph inputs – they are baked into the model as constants.
            init = _copy.deepcopy(constant_tensors[name])
            new_initializers.append(init)
            added_initializer_names.add(name)
        elif name in orig_inputs:
            new_inputs.append(_copy.deepcopy(orig_inputs[name]))
        elif name in orig_value_info:
            new_inputs.append(_copy.deepcopy(orig_value_info[name]))
        elif name in orig_initializers:
            init = orig_initializers[name]
            new_initializers.append(_copy.deepcopy(init))
            added_initializer_names.add(name)
            new_inputs.append(_make_value_info_from_initializer(init))
        elif name in orig_outputs:
            new_inputs.append(_copy.deepcopy(orig_outputs[name]))
        else:
            new_inputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, []))

    # Resolve dynamic shapes for external inputs.  With ONNX shape inference
    # run on the full model first, most shapes are already static in value_info.
    # The fallback below only handles Reshape/Transpose/Expand where ONNX
    # leaves dynamic dims (unk__*).
    shape_cache = {}
    for vi in new_inputs:
        tt = vi.type.tensor_type
        # Resolve shape if empty (no dims) or any dim is dynamic (0 without param or symbolic)
        has_dynamic = not tt.shape.dim or any(
            (d.dim_value == 0 and not d.dim_param) or d.dim_param
            for d in tt.shape.dim
        )
        if has_dynamic:
            resolved = _resolve_tensor_shape(
                vi.name, all_nodes, orig_value_info, orig_inputs,
                orig_outputs, orig_initializers, constant_tensors,
                shape_cache,
            )
            if resolved is not None:
                _set_static_dims(vi, resolved)
                if log_prefix:
                    print(f'{log_prefix}: resolved shape for {vi.name} -> {resolved}')

    for name in sorted((external_inputs | used)):
        if name in orig_initializers and name not in added_initializer_names:
            new_initializers.append(_copy.deepcopy(orig_initializers[name]))

    for name in sorted(external_outputs):
        matched = False
        for out in graph.output:
            if out.name == name:
                new_outputs.append(_copy.deepcopy(out))
                matched = True
                break
        if not matched and name in orig_value_info:
            new_outputs.append(_copy.deepcopy(orig_value_info[name]))
        elif not matched and name in orig_initializers:
            new_outputs.append(_make_value_info_from_initializer(orig_initializers[name]))
        elif not matched:
            new_outputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, []))

    # Resolve dynamic shapes for external outputs too (needed for ONNX→MLIR
    # import which uses output shapes for type inference).
    for vi in new_outputs:
        tt = vi.type.tensor_type
        has_dynamic = not tt.shape.dim or any(
            (d.dim_value == 0 and not d.dim_param) or d.dim_param
            for d in tt.shape.dim
        )
        if has_dynamic:
            resolved = _resolve_tensor_shape(
                vi.name, all_nodes, orig_value_info, orig_inputs,
                orig_outputs, orig_initializers, constant_tensors,
                shape_cache,
            )
            if resolved is not None:
                _set_static_dims(vi, resolved)
                if log_prefix:
                    print(f'{log_prefix}: resolved output shape for {vi.name} -> {resolved}')

    new_value_info = []
    for name in sorted(produced | used):
        if name in orig_value_info:
            new_value_info.append(_copy.deepcopy(orig_value_info[name]))

    # Deep-copy nodes
    nodes_copy = [_copy.deepcopy(n) for n in nodes_subset]

    # Ensure initializer names are unique
    name_map = {}
    existing_names = set()
    for n in nodes_copy:
        existing_names.update([s for s in n.input if s])
        existing_names.update([s for s in n.output if s])
    existing_names.update([vi.name for vi in new_inputs])
    existing_names.update([vi.name for vi in new_outputs])
    existing_names.update([vi.name for vi in new_value_info])

    for idx_init, init in enumerate(new_initializers):
        base = init.name
        candidate = init_name_fn(base, idx_init)
        i = 0
        while candidate in existing_names:
            i += 1
            candidate = f"{candidate}_{i}"
        name_map[base] = candidate
        existing_names.add(candidate)
        init.name = candidate

    if name_map:
        for n in nodes_copy:
            for ii, v in enumerate(list(n.input)):
                if v in name_map:
                    n.input[ii] = name_map[v]
        for vi in new_inputs:
            if vi.name in name_map:
                vi.name = name_map[vi.name]
        for vi in new_outputs:
            if vi.name in name_map:
                vi.name = name_map[vi.name]
        for vi in new_value_info:
            if vi.name in name_map:
                vi.name = name_map[vi.name]

    # Create new graph and model
    new_graph = helper.make_graph(
        nodes_copy,
        graph_name,
        inputs=new_inputs,
        outputs=new_outputs,
        initializer=new_initializers
    )
    if new_value_info:
        new_graph.value_info.extend(new_value_info)

    new_model = helper.make_model(new_graph)
    new_model.ir_version = model.ir_version
    new_model.opset_import.extend(model.opset_import)

    try:
        inferred = shape_inference.infer_shapes(new_model)

        # Propagate static output shapes back into value_info.
        # shape_inference may leave value_info entries with dynamic dims
        # even when the matching output has a concrete shape.
        output_shapes = {}
        for out in inferred.graph.output:
            dims = _get_static_shape(out.type.tensor_type)
            if dims is not None:
                output_shapes[out.name] = dims

        if output_shapes:
            for vi in inferred.graph.value_info:
                shape = output_shapes.get(vi.name)
                if shape is not None and _has_dynamic_dims(vi.type.tensor_type):
                    _set_static_dims(vi, shape)

        onnx.checker.check_model(inferred)
        return inferred
    except Exception as e:
        if log_prefix:
            print(f'{log_prefix}: inference/check failed: {e}; using raw model')
        return _copy.deepcopy(new_model)

# ---- Layer cache (disk-backed) ----------------------------------------
#
# Caches the output of generate_onnx_layers_from_model() under
# .pytest_cache/d/onnx_layer_cache/<model>/ so repeated pytest runs skip
# shape inference + extraction and just onnx.load each layer file.
#
# Layout:
#   <cache_dir>/<model>/
#       manifest.json     -- ordered list of layers + metadata
#       full_model.onnx   -- shape-inferred full model
#       <safe_layer_name>.onnx ...

_LAYER_CACHE_VERSION = 1  # bump when extraction logic changes

def _load_layers_from_cache(key_dir: Path, model_file: str,
                            node_groups, dedup):
    """Return (full_model, layers_dict) or None on miss / mismatch / corrupt."""
    manifest_path = key_dir / "manifest.json"
    full_model_path = key_dir / "full_model.onnx"

    if not manifest_path.exists() or not full_model_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())

        # Invalidate if the model file or extraction options changed.
        stat = Path(model_file).stat()
        if (manifest.get("version") != _LAYER_CACHE_VERSION
                or manifest.get("model_size") != stat.st_size
                or manifest.get("model_mtime_ns") != stat.st_mtime_ns
                or manifest.get("node_groups") != node_groups
                or manifest.get("dedup") != dedup):
            return None
        
        layers = {}

        for entry in manifest["layers"]:
            layer_path = key_dir / entry["filename"]
            if not layer_path.exists():
                return None  # partial cache, treat as miss

            layers[entry["name"]] = ModelWithMetadata(
                node_index=entry.get("node_index"), path=layer_path)
        return ModelWithMetadata(path=full_model_path), layers
    
    except Exception as e:
        print(f"[onnx-layer-cache] ignoring corrupt cache at {key_dir}: {e}")
        return None


def _save_layers_to_cache(key_dir: Path, full_model, layers: dict,
                          model_file: str, node_groups, dedup) -> None:
    """Atomically write full_model + layers + manifest under key_dir."""
    import os
    import shutil

    tmp_dir = key_dir.with_name(key_dir.name + f".tmp.{os.getpid()}")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=False)

    try:
        onnx.save(full_model, str(tmp_dir / "full_model.onnx"))

        entries = []
        used = set()
        for name, layer in layers.items():
            model = layer.model if hasattr(layer, "model") else layer
            node_index = getattr(layer, "node_index", None)

            filename = f"{name}.onnx"
            i = 0
            while filename in used:
                i += 1
                filename = f"{name}_{i}.onnx"
            used.add(filename)

            onnx.save(model, str(tmp_dir / filename))
            entries.append({
                "name": name,
                "filename": filename,
                "node_index": node_index,
            })

        stat = Path(model_file).stat()
        (tmp_dir / "manifest.json").write_text(json.dumps({
            "version": _LAYER_CACHE_VERSION,
            "model_size": stat.st_size,
            "model_mtime_ns": stat.st_mtime_ns,
            "node_groups": node_groups,
            "dedup": dedup,
            "layers": entries,
        }, indent=2))

        if key_dir.exists():
            shutil.rmtree(key_dir)
        tmp_dir.rename(key_dir)
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise

# ---- Layer / subgraph extraction ----

def generate_onnx_layers_from_model(model, node_groups=None, dedup=True):
    # Run shape inference on the full model so value_info is populated.
    # This lets _fold_Shape resolve shapes from value_info and eliminates
    # the need for most per-op handlers in _resolve_tensor_shape.
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass

    existing_cases = set()
    layer_configs = {}

    graph = model.graph
    nodes = graph.node
    node_count = len(nodes)

    def match_group(start_index):
        if node_groups is None:
            return 0
        for group in node_groups:
            if start_index + len(group) <= node_count:
                if all(nodes[start_index + i].op_type == group[i] for i in range(len(group))):
                    return len(group)
        return 0

    index = 0
    part_num = 0
    # Track the original node index for matching with full model MLIR.
    # This counts only non-Constant nodes since those are what torch-mlir converts.
    original_node_index = 0
    while index < node_count:

        if (nodes[index].op_type.lower() == 'constant'):
            index += 1
            continue

        # Record the original index BEFORE incrementing for this layer
        start_node_index = original_node_index

        group_size = match_group(index)
        if group_size > 0:
            part_nodes = nodes[index:index + group_size]
            part_node_indices = list(range(index, index + group_size))
            index += group_size
            original_node_index += group_size
        else:
            part_nodes = [nodes[index]]
            part_node_indices = [index]
            index += 1
            original_node_index += 1

        source_node_names = [
            _source_node_name(node, node_index)
            for node_index, node in zip(part_node_indices, part_nodes)
        ]
        node_name = " -> ".join(source_node_names)

        layer_name = f"layer_{part_nodes[0].op_type}_{part_num}"
        part_num += 1

        final_model = _build_model_from_node_subset(
            part_nodes,
            nodes,
            model,
            graph_name=f'part{part_num - 1}_graph',
            init_name_fn=lambda base, idx, pn=part_num - 1: f"{base}_part{pn}_init{idx}",
            log_prefix=f'layer {layer_name}',
        )

        # Propagate inferred static output shapes back into the original model's
        # value_info so subsequent extractions see static shapes for intermediate
        # tensors produced by earlier layers.
        for out in final_model.graph.output:
            tt = out.type.tensor_type
            if _has_dynamic_dims(tt):
                continue
            shape = [d.dim_value for d in tt.shape.dim]
            vi = next((vi for vi in graph.value_info if vi.name == out.name), None)
            if not _should_propagate_layer_shape(vi, shape):
                continue
            _set_static_dims(vi, shape) if vi else graph.value_info.append(_copy.deepcopy(out))

        # Fix dynamic batch dimensions to static 1 for layer testing
        _fix_batch_dimension_to_one(final_model)

        # check duplication: too heavy to compare model, just compare signature
        if dedup:
            m_signature_json = json.dumps(model_signature(final_model))
            found_existing = False
            for c in existing_cases:
                if c == m_signature_json:
                    found_existing = True
                    break

            if not found_existing:
                existing_cases.add(m_signature_json)
            else:
                continue

        # Wrap model with metadata (node_index for matching with full model MLIR)
        layer_configs[layer_name] = ModelWithMetadata(final_model, start_node_index)

    return layer_configs


def extract_onnx_subgraph(model, from_index, to_index):
    """
    Extract a subgraph from an ONNX model given start and end node indices.

    The indices refer to non-Constant nodes in the model graph (same indexing
    as used in executor discovery JSON's _node_index field).

    Args:
        model: ONNX model (ModelProto or ModelWithMetadata)
        from_index: Start node index (inclusive, refers to non-Constant nodes)
        to_index: End node index (inclusive, refers to non-Constant nodes)

    Returns:
        ModelWithMetadata containing the extracted subgraph
    """
    # Unwrap ModelWithMetadata if needed
    if hasattr(model, 'model'):
        model = model.model

    # Run shape inference on the full model so value_info is populated.
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass

    graph = model.graph
    nodes = list(graph.node)

    # Filter out Constant nodes to get the same indexing as layer extraction
    non_constant_nodes = []
    non_constant_indices = []  # Maps from filtered index -> original index
    for i, node in enumerate(nodes):
        if node.op_type.lower() != 'constant':
            non_constant_nodes.append(node)
            non_constant_indices.append(i)

    total_non_constant = len(non_constant_nodes)

    # Validate indices
    if from_index < 0 or from_index >= total_non_constant:
        raise ValueError(f"from_index {from_index} out of range [0, {total_non_constant})")
    if to_index < 0 or to_index >= total_non_constant:
        raise ValueError(f"to_index {to_index} out of range [0, {total_non_constant})")
    if from_index > to_index:
        raise ValueError(f"from_index {from_index} must be <= to_index {to_index}")

    # Map to original node indices
    original_from_idx = non_constant_indices[from_index]
    original_to_idx = non_constant_indices[to_index]

    # Get all nodes in range (including any Constant nodes between them)
    subgraph_nodes = nodes[original_from_idx:original_to_idx + 1]

    final_model = _build_model_from_node_subset(
        subgraph_nodes,
        nodes,
        model,
        graph_name=f"subgraph_{from_index}_to_{to_index}",
        init_name_fn=lambda base, idx: f"{base}_subgraph_init{idx}",
        log_prefix="[Subgraph]",
    )

    # Fix dynamic batch dimensions to static 1 for subgraph testing
    _fix_batch_dimension_to_one(final_model)

    return ModelWithMetadata(final_model, from_index)


def _build_onnx_layer_cases(model_prefix: str, model, layers):
    cases = [
        OnnxLayerCase(
            name=f"{model_prefix}_{key}",
            data=layer,
        )
        for key, layer in layers.items()
    ]
    cases.append(OnnxLayerCase(name=f"{model_prefix}_full_model", data=model, is_full_model=True))
    return cases

def _load_cached_layers(cache, model_file: str, name_stem: str,
                                    node_groups, dedup):
    """Shared cache-aware wrapper used by both _from_file and _from_hf."""

    cache_dir = cache.mkdir('onnx_layer_cache')
    key_dir = cache_dir / name_stem
    cached = _load_layers_from_cache(key_dir, model_file, node_groups, dedup)

    if cached is not None:
        full_model, layers = cached
        print(f"[onnx-layer-cache] HIT  {name_stem} -> {key_dir}")
        return _build_onnx_layer_cases(name_stem, full_model, layers)

    print(f"[onnx-layer-cache] MISS {name_stem} -> {key_dir}")
    full_model = get_full_model(model_file)
    layers = generate_onnx_layers_from_model(full_model, node_groups, dedup)
    try:
        _save_layers_to_cache(key_dir, full_model, layers,
                                model_file, node_groups, dedup)
    except Exception as e:
        print(f"[onnx-layer-cache] save failed (continuing): {e}")
    return _build_onnx_layer_cases(name_stem, full_model, layers)


def generate_onnx_layers_from_hf(cache, repo_id, filename, node_groups=None, dedup=True):
    model_file = get_hf_model_file(cache, repo_id, filename)
    return _load_cached_layers(
        cache, model_file, Path(filename).stem, node_groups, dedup)


def generate_onnx_layers_from_file(cache, filepath: Path, node_groups=None, dedup=False):
    return _load_cached_layers(
        cache, str(filepath), filepath.stem, node_groups, dedup)

# ---- Pytest fixtures ----

@pytest.fixture
def onnx_model(request, case_config):
    return request.getfixturevalue(case_config['onnx_model'])


def onnx_model_fixture(fun):
    """Decorator for fixtures that build an in-memory onnx.ModelProto (mirror of
    tensorflow.keras_model_fixture). The wrapped fixture is consumed via the
    existing onnx_model -> onnx_model_file -> onnx_mlir_model_file chain, so a
    parametric builder needs no new downstream plumbing."""
    @versioned_unhashable_object_fixture
    @wraps(fun)
    def wrapper(**kwargs):
        request = kwargs.get("request")
        if request is None:
            raise ValueError("onnx_model_fixture requires a request fixture")
        record_property = request.getfixturevalue("record_property")
        record_property("compiler_input", f"onnx:{fun.__module__}.{fun.__qualname__}")
        return fun(**kwargs)

    return wrapper


@versioned_generated_file_fixture("onnx")
def onnx_model_file(request, versioned_file, onnx_model):
    onnx.checker.check_model(onnx_model)
    onnx.save(onnx_model, versioned_file)


def is_model_bf16(model: onnx.ModelProto) -> bool:
    """Check if model is already in BF16 format."""
    return any(
        init.data_type == TensorProto.BFLOAT16
        for init in model.graph.initializer
    )


@versioned_hashable_object_fixture
def onnx_bf16_config(request):
    """Return BF16 conversion config for version hashing.

    This ensures cache invalidation when --auto-convert-bf16 flag changes.
    """
    return {
        "auto_convert_bf16": request.config.getoption("--auto-convert-bf16", default=False)
    }


@versioned_generated_file_fixture("onnx_bf16")
def onnx_bf16_model_file(request, versioned_file, onnx_model_file, onnx_bf16_config):
    """Convert FP32 ONNX to BF16 if --auto-convert-bf16 is enabled.

    Always saves to versioned_file location (the decorator always returns versioned_file).

    Note: onnx_model_file is a Path (versioned_generated_file_fixture unwraps VersionedFile).
    """
    import shutil
    use_bf16 = request.config.getoption("--auto-convert-bf16", default=False)

    if not use_bf16:
        # No conversion - copy original model to versioned location
        print(f"[BF16] Conversion disabled, copying original model to {versioned_file}")
        shutil.copy(str(onnx_model_file), str(versioned_file))
        return versioned_file

    # Load the model
    model = onnx.load(str(onnx_model_file))

    # Check if already BF16
    if is_model_bf16(model):
        print(f"[BF16] Model already in BF16 format, copying to {versioned_file}")
        shutil.copy(str(onnx_model_file), str(versioned_file))
        return versioned_file

    # Convert to BF16
    print(f"[BF16] Converting {onnx_model_file.name} to BF16...")
    converted_model = convert_fp32_to_bf16(model)

    # Save converted model
    onnx.save(converted_model, str(versioned_file))
    print(f"[BF16] Saved to: {versioned_file}")

    return versioned_file


@versioned_generated_file_fixture("mlir")
def onnx_mlir_model_file(request, versioned_file, onnx_model_file, onnx_bf16_model_file, onnx_bf16_config):
    """Convert ONNX model to MLIR with enhanced error diagnostics.

    Uses BF16 model if --auto-convert-bf16 is enabled, otherwise uses original model.
    This ensures the compiler receives the correctly converted model based on user options.

    Note: Both onnx_model_file and onnx_bf16_model_file are Path objects
    (versioned_generated_file_fixture unwraps VersionedFile to Path).
    """
    use_bf16 = request.config.getoption("--auto-convert-bf16", default=False)
    model_path = onnx_bf16_model_file if use_bf16 else onnx_model_file

    if use_bf16:
        print(f"[BF16] Using BF16 model for MLIR conversion: {model_path}")

    # Pre-validation: verify ONNX model integrity
    try:
        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)
        print(f"[MLIR] ONNX model validation passed: {model_path}")
    except Exception as e:
        raise RuntimeError(
            f"ONNX model validation failed for {model_path}\n"
            f"Error: {e}"
        )

    # Attempt ONNX to MLIR conversion with comprehensive error handling
    try:
        result = subprocess.run(
            [sys.executable, "-m", "iree.compiler.tools.import_onnx",
             str(model_path), "-o", str(versioned_file), "--data-prop"],
            capture_output=True, text=True, timeout=300
        )

        if result.returncode != 0:
            # Provide full diagnostic information
            error_msg = f"iree.compiler.tools.import_onnx failed for {model_path}\n"
            error_msg += f"Return code: {result.returncode}\n"
            error_msg += f"stdout:\n{result.stdout or '(empty)'}\n"
            error_msg += f"stderr:\n{result.stderr or '(empty)'}\n"
            error_msg += f"Model file size: {model_path.stat().st_size if model_path.exists() else 'N/A'} bytes"
            raise RuntimeError(error_msg)

        print(f"[MLIR] Successfully converted {model_path} to {versioned_file}")

    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"iree.compiler.tools.import_onnx timed out for {model_path}\n"
            "This may indicate the model is too large or complex for the current environment."
        )
    except Exception as e:
        raise RuntimeError(
            f"iree.compiler.tools.import_onnx failed with exception for {model_path}\n"
            f"Error: {type(e).__name__}: {e}"
        )


@versioned_hashable_object_fixture
def onnx_params(case_config):
    return {"opset": case_config.get("opset", 20)}


@versioned_cached_data_fixture
def onnx_reference_results(request, onnx_model_file):
    onnx_model = onnx.load(str(onnx_model_file))
    onnx.checker.check_model(onnx_model)
    ort_session = onnxruntime.InferenceSession(str(onnx_model_file))
    ort_inputs = {ort_session.get_inputs()[0].name: sample_input.numpy()}
    ort_outs = ort_session.run(None, ort_inputs)
    return ort_outs



# ---- Model loading ----

def get_full_model(model_file):

    model = onnx.load(model_file)
    # Run shape inference on the original model to get value_info with shapes
    inferred_model = None
    try:
        inferred_model = shape_inference.infer_shapes(model)
    except Exception as e:
        inferred_model = model

    return inferred_model



# ---- BF16 conversion ----

def _float32_to_bfloat16(arr: np.ndarray) -> np.ndarray:
    """Convert float32 numpy array to bfloat16 (stored as uint16)."""
    arr_uint32 = arr.view(np.uint32)
    arr_bf16 = (arr_uint32 >> 16).astype(np.uint16)
    return arr_bf16


def _fix_batch_dimension_to_one(model: onnx.ModelProto) -> int:
    """Fix dynamic batch dimensions (?, -1) to 1 for all inputs, outputs, and value_info."""
    modified_count = 0
    for value_info in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        tensor_type = value_info.type.tensor_type
        if tensor_type.HasField('shape') and len(tensor_type.shape.dim) > 0:
            first_dim = tensor_type.shape.dim[0]
            if first_dim.HasField('dim_param'):
                first_dim.ClearField('dim_param')
                first_dim.dim_value = 1
                modified_count += 1
            elif not first_dim.HasField('dim_value'):
                first_dim.dim_value = 1
                modified_count += 1
    return modified_count


def convert_fp32_to_bf16(model: onnx.ModelProto) -> onnx.ModelProto:
    """Convert FP32 ONNX model to BF16 format.

    Returns the converted model and prints accuracy metrics.
    """
    import copy
    model = copy.deepcopy(model)  # Don't modify original

    # Fix batch dimension
    batch_fixed = _fix_batch_dimension_to_one(model)
    if batch_fixed > 0:
        print(f"[BF16] Fixed {batch_fixed} input(s) to have batch=1")

    total_count = 0
    max_error = 0.0

    # Convert initializers (weights)
    for init in model.graph.initializer:
        if init.data_type != TensorProto.FLOAT:
            continue

        # Get FP32 data
        if init.raw_data:
            fp32_data = np.frombuffer(init.raw_data, dtype=np.float32).copy()
        elif init.float_data:
            fp32_data = np.array(init.float_data, dtype=np.float32)
        else:
            continue

        # Convert to BF16
        bf16_data = _float32_to_bfloat16(fp32_data)

        # Track error
        total_count += len(fp32_data)
        fp32_back = (bf16_data.astype(np.uint32) << 16).view(np.float32)
        max_error = max(max_error, np.max(np.abs(fp32_data - fp32_back)))

        # Replace data
        init.raw_data = bf16_data.tobytes()
        init.float_data[:] = []
        init.data_type = TensorProto.BFLOAT16

    # Run shape inference
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass

    # Update input/output types
    for value_info in list(model.graph.input) + list(model.graph.output):
        if value_info.type.tensor_type.elem_type == TensorProto.FLOAT:
            value_info.type.tensor_type.elem_type = TensorProto.BFLOAT16

    # Update intermediate value_info types
    for value_info in model.graph.value_info:
        if value_info.type.tensor_type.elem_type == TensorProto.FLOAT:
            value_info.type.tensor_type.elem_type = TensorProto.BFLOAT16

    print(f"[BF16] Converted {total_count} weight values, max error: {max_error:.6f}")
    return model


@pytest.fixture
def onnx_layer_model(request):
    case_name = request.param.name
    model_data = request.param.data
    version = "onnx_layer_model_" + case_name

    if isinstance(request.param, OnnxLayerCase):
        if request.config.getoption("--onnx-print-layer-info") and not request.param.is_full_model:
            print(_format_layer_node_name(request.param))

    record_property = request.getfixturevalue("record_property")
    record_property("compiler_input", f"onnx:{case_name}")

    return VersionedUncachedData(data=model_data, version=version)


# Re-export numpy executor functions and fixture for backward compatibility
# These are defined in .numpy to avoid circular dependencies with .torch
from .numpy import (
    _has_bf16_matmul,
    _has_bf16_einsum,
    _has_gelu,
    _numpy_maxpool,
    _execute_onnx_model_numpy,
    numpy_gelu_reference_results,
    numpy_reference_results,
)


@versioned_unhashable_object_fixture
def composite_reference_results(request, input_data):
    """
    Generate reference using a chained fallback strategy:
    1. ONNXRuntime (fastest, most accurate for f32)
    2. numpy fallback (for bf16 MatMul/Einsum/MaxPool)
    3. llvmcpu fallback (IREE reference compilation)
    4. torch fallback (last resort for bf16 models with unsupported ops)
    """
    # Try ONNX-based paths first if an ONNX model is available.
    try:
        onnx_model_file = request.getfixturevalue("onnx_model_file")
        onnx_model = onnx.load(str(onnx_model_file))

        # 1. Try ONNXRuntime first
        if not _has_bf16_matmul(onnx_model) or not _has_bf16_einsum(onnx_model):
            try:
                ort_session = onnxruntime.InferenceSession(str(onnx_model_file))
                ort_inputs = {inp.name: input_data[i] for i, inp in enumerate(ort_session.get_inputs())}
                return ort_session.run(None, ort_inputs)
            except Exception:
                pass

        # 2. Try numpy fallback
        try:
            return _execute_onnx_model_numpy(onnx_model, input_data)
        except Exception:
            pass
    except Exception:
        pass

    # 3. Try llvmcpu fallback
    try:
        return request.getfixturevalue("llvmcpu_reference_results").data
    except Exception:
        pass

    # 4. Last resort: torch fallback
    print("Warning: All previous references (ONNXRuntime, numpy, llvmcpu) failed, falling back to torch reference")
    return request.getfixturevalue("torch_reference_results").data
