import importlib.util
import json
import numpy as np
import os
import sys
import onnx
import onnxruntime
import pytest
import re
import shutil
from dataclasses import dataclass
from filelock import FileLock
from pathlib import Path
from onnx import helper, numpy_helper, TensorProto
try:
    import torch
    import torch.nn.functional as F
except ImportError:
    print("Warning: pytorch not available")
    torch = None
    F = None

from .cache_utils import atomic_write_json_manifest
from .cases import Case
from .versioned_fixtures import (
    versioned_unhashable_object_fixture,
    versioned_generated_file_fixture,
    VersionedUncachedData,
)

if torch:
    TORCH_DTYPE_MAP = {
        "float16": torch.float16, "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32, "float": torch.float32,
        "int8": torch.int8, "int16": torch.int16,
        "int32": torch.int32, "int": torch.int32,
        "int64": torch.int64, "long": torch.int64,
        "uint8": torch.uint8,
    }


_TORCH_LAYER_CACHE_VERSION = 1

# Registry for lazy model loaders (used by tests that avoid keeping the model
# object in every pytest-xdist worker during collection, e.g. torchvision zoo
# models).  The loader callable takes no arguments and returns a torch.nn.Module.
_TORCH_MODEL_LOADERS = {}
_TORCH_MODEL_CACHE = {}


def register_torch_model_loader(key: str, loader):
    """Register a lazy model loader that fixtures can use to load a model on demand.

    Args:
        key: Unique identifier for this model (used in case data).
        loader: Callable that returns a torch.nn.Module.
    """
    if key in _TORCH_MODEL_LOADERS:
        raise ValueError(f"Torch model loader already registered for key: {key}")
    _TORCH_MODEL_LOADERS[key] = loader


def _load_registered_model_cached(key: str):
    """Call a registered model loader and cache the result per worker process."""
    if key not in _TORCH_MODEL_CACHE:
        loader = _TORCH_MODEL_LOADERS.get(key)
        if loader is None:
            raise ValueError(f"No torch model loader registered for key: {key}")
        model = loader()
        if not isinstance(model, torch.nn.Module):
            raise ValueError(f"Loader for {key} must return a torch.nn.Module, got {type(model)}")
        _TORCH_MODEL_CACHE[key] = model
    return _TORCH_MODEL_CACHE[key]


def _get_model_from_case_config(case_config):
    """Resolve a torch.nn.Module from case_config, supporting several storage modes.

    Supported keys (checked in order):
      - "model": an already-loaded torch.nn.Module.
      - "model_path": path to a .pt/.pth/.py model file.
      - "model_loader_key": key registered with register_torch_model_loader().
    """
    model = case_config.get("model")
    if model is not None:
        return model

    model_path = case_config.get("model_path")
    if model_path:
        if model_path not in _TORCH_MODEL_CACHE:
            _TORCH_MODEL_CACHE[model_path], _ = _load_torch_model_and_inputs(Path(model_path), None)
        return _TORCH_MODEL_CACHE[model_path]

    loader_key = case_config.get("model_loader_key")
    if loader_key:
        return _load_registered_model_cached(loader_key)

    raise ValueError(
        "case_config must contain one of 'model', 'model_path', or 'model_loader_key'"
    )


def _parse_torch_mlir_op(mlir_path: Path):
    """Parse a torch MLIR file containing torch.operator 'onnx.*' ops.

    Returns (op_type, attrs, input_count) where attrs is a dict of
    ONNX-style attribute names to Python values.
    """
    content = mlir_path.read_text()

    # Find torch.operator "onnx.*" ops
    op_pattern = r'torch\.operator\s+"onnx\.([^"]+)"\s*\(([^)]+)\)\s*\{([^}]+)\}'
    op_matches = re.findall(op_pattern, content, re.DOTALL)
    if not op_matches:
        raise ValueError("No torch.operator onnx ops found in MLIR")

    onnx_op_type = op_matches[0][0]
    attr_text = op_matches[0][2]

    # Parse attributes
    attrs = {}
    # Arrays may contain commas, so match brackets as a group.
    attr_pattern = r'torch\.onnx\.([\w_]+)\s*=\s*(\[[^\]]*\]|[^,\n]+)'
    for match in re.finditer(attr_pattern, attr_text):
        name = match.group(1)
        value_str = match.group(2).strip()
        if value_str.startswith('[') and value_str.endswith(']'):
            items = re.findall(r'([-\d]+)\s*:\s*\w+', value_str)
            attrs[name] = [int(x) for x in items]
        else:
            m = re.match(r'([-\d]+)\s*:\s*\w+', value_str)
            if m:
                attrs[name] = int(m.group(1))
            else:
                attrs[name] = value_str

    # Count inputs from the operand list
    input_count = len([x for x in op_matches[0][1].split(',') if x.strip()])

    return onnx_op_type, attrs, input_count


def _parse_topk_k_from_mlir(mlir_path: Path, axis: int, input_shape: list):
    """Infer the 'k' value for a TopK op from the torch-mlir file.

    Tries multiple strategies:
      1. Parse an inline dense<k> constant.
      2. Extract the output shape from the TopK result type.
      3. Fall back to the function return type.
    """
    content = mlir_path.read_text()

    # Strategy 1: inline dense constant (e.g. dense<300>)
    dense_match = re.search(r'dense<(\d+)>\s*:\s*tensor<1xsi64>', content)
    if dense_match:
        return int(dense_match.group(1))

    # Strategy 2: output shape on the TopK op itself
    topk_match = re.search(
        r'torch\.operator\s+"onnx\.TopK"\s*\([^)]*\)\s*\{[^}]*\}\s*:\s*\([^)]*\)\s*->\s*\(([^)]+)\)',
        content,
    )
    if topk_match:
        types_str = topk_match.group(1)
        shape_match = re.search(r'!torch\.vtensor<\[([^\]]+)\]', types_str)
        if shape_match:
            dims = [int(d.strip()) for d in shape_match.group(1).split(',')]
            axis_idx = axis if axis >= 0 else len(input_shape) + axis
            if 0 <= axis_idx < len(dims):
                return dims[axis_idx]

    # Strategy 3: function return type
    ret_match = re.search(r'->\s*\(?\s*!torch\.vtensor<\[([^\]]+)\]', content)
    if ret_match:
        dims = [int(d.strip()) for d in ret_match.group(1).split(',')]
        axis_idx = axis if axis >= 0 else len(input_shape) + axis
        if 0 <= axis_idx < len(dims):
            return dims[axis_idx]

    raise ValueError(f"Could not infer k for TopK from {mlir_path}")


def _execute_torch_mlir(mlir_path: Path, input_data):
    """Execute a torch MLIR file directly using PyTorch.

    Parses torch.operator 'onnx.*' ops and dispatches to the corresponding PyTorch function
    """
    op_type, attrs, input_count = _parse_torch_mlir_op(mlir_path)

    if op_type == 'MaxPool':
        x = _np_array_to_torch(input_data[0])
        kernel_shape = attrs.get('kernel_shape', [2, 2])
        strides = attrs.get('strides', kernel_shape)
        pads = attrs.get('pads', [0] * (2 * len(kernel_shape)))
        ceil_mode = attrs.get('ceil_mode', 0)
        ndim = len(kernel_shape)

        # Handle padding manually because ONNX supports asymmetric padding
        if any(p != 0 for p in pads):
            pad_begin = pads[:ndim]
            pad_end = pads[ndim:]
            torch_pad = []
            for i in range(ndim - 1, -1, -1):
                torch_pad.extend([pad_begin[i], pad_end[i]])
            x = F.pad(x, torch_pad, value=-float('inf'))

        pool_kwargs = dict(kernel_size=kernel_shape, stride=strides, padding=0, ceil_mode=bool(ceil_mode))
        if ndim == 1:
            result = F.max_pool1d(x, **pool_kwargs)
        elif ndim == 2:
            result = F.max_pool2d(x, **pool_kwargs)
        elif ndim == 3:
            result = F.max_pool3d(x, **pool_kwargs)
        else:
            raise ValueError(f"Unsupported MaxPool spatial rank: {ndim}")
        return [_torch_tensor_to_np(result)]

    elif op_type == 'Conv':
        if len(input_data) < 2:
            raise ValueError(
                f"Conv expects at least 2 inputs (data, weight), got {len(input_data)}. "
                f"This usually means _execute_torch_mlir was called with a full model "
                f"instead of a single-op MLIR file."
            )
        x = _np_array_to_torch(input_data[0])
        weight = _np_array_to_torch(input_data[1])
        bias = _np_array_to_torch(input_data[2]) if len(input_data) > 2 else None

        # Determine spatial rank from weight shape: [M, C/group, *kernel_shape]
        ndim = weight.ndim - 2
        if ndim < 1 or ndim > 3:
            raise ValueError(f"Unsupported Conv spatial rank: {ndim}")

        kernel_shape = attrs.get('kernel_shape', list(weight.shape[2:]))
        strides = attrs.get('strides', [1] * ndim)
        dilations = attrs.get('dilations', [1] * ndim)
        groups = attrs.get('group', 1)
        pads = attrs.get('pads', [0] * (2 * ndim))

        # ONNX pads are [begin1, begin2, ..., end1, end2, ...]
        # PyTorch only supports symmetric padding for conv, so handle asymmetric manually.
        pad_begin = pads[:ndim]
        pad_end = pads[ndim:]
        symmetric_pads = [(b, e) for b, e in zip(pad_begin, pad_end)]
        needs_explicit_pad = any(b != e for b, e in symmetric_pads)

        if needs_explicit_pad:
            torch_pad = []
            for i in range(ndim - 1, -1, -1):
                torch_pad.extend([pad_begin[i], pad_end[i]])
            x = F.pad(x, torch_pad)
            padding = [0] * ndim
        else:
            padding = pad_begin

        conv_kwargs = dict(
            stride=strides,
            padding=padding,
            dilation=dilations,
            groups=groups,
            bias=bias is not None,
        )
        if bias is not None:
            conv_kwargs['bias'] = bias

        if ndim == 1:
            result = F.conv1d(x, weight, **conv_kwargs)
        elif ndim == 2:
            result = F.conv2d(x, weight, **conv_kwargs)
        elif ndim == 3:
            result = F.conv3d(x, weight, **conv_kwargs)

        return [_torch_tensor_to_np(result)]

    elif op_type == 'Cast':
        x = _np_array_to_torch(input_data[0])
        to_dtype = attrs.get('to', 1)
        result = x.to(_onnx_dtype_to_torch(to_dtype))
        return [_torch_tensor_to_np(result)]

    elif op_type == 'TopK':
        x = _np_array_to_torch(input_data[0])
        axis = attrs.get('axis', -1)
        largest = attrs.get('largest', 1)
        sorted_attr = attrs.get('sorted', 1)
        k = _parse_topk_k_from_mlir(mlir_path, axis, list(x.shape))
        values, indices = torch.topk(x, k, dim=axis, largest=bool(largest), sorted=bool(sorted_attr))
        # Return only the outputs that the function actually returns
        content = mlir_path.read_text()
        ret_match = re.search(r'return\s+([^:]+)\s*:', content)
        if ret_match:
            ret_expr = ret_match.group(1)
            outputs = []
            if '#0' in ret_expr:
                outputs.append(_torch_tensor_to_np(values))
            if '#1' in ret_expr:
                outputs.append(_torch_tensor_to_np(indices))
            if outputs:
                return outputs
        return [_torch_tensor_to_np(values), _torch_tensor_to_np(indices)]

    else:
        raise ValueError(f"torch MLIR direct execution not yet implemented for op: {op_type}")


def _np_array_to_torch(arr):
    """Convert numpy array to torch tensor, handling bfloat16 dtype."""
    if str(arr.dtype) == 'bfloat16':
        return torch.from_numpy(arr.view(np.uint16)).view(torch.bfloat16)
    return torch.from_numpy(arr)


def _torch_tensor_to_np(tensor):
    """Convert torch tensor to numpy array, handling bfloat16 dtype."""
    if tensor.dtype == torch.bfloat16:
        arr = tensor.view(torch.uint16).numpy()
        try:
            return arr.view('bfloat16')
        except TypeError:
            import ml_dtypes
            return arr.view(ml_dtypes.bfloat16)
    return tensor.numpy()


# ONNX TensorProto data type -> torch dtype mapping for Cast support
_ONNX_TO_TORCH_DTYPE = {
    1: 'float32',    # TensorProto.FLOAT
    2: 'uint8',      # TensorProto.UINT8
    3: 'int8',       # TensorProto.INT8
    5: 'int16',      # TensorProto.INT16
    6: 'int32',      # TensorProto.INT32
    7: 'int64',      # TensorProto.INT64
    9: 'bool',       # TensorProto.BOOL
    10: 'float16',   # TensorProto.FLOAT16
    11: 'float64',   # TensorProto.DOUBLE
    16: 'bfloat16',  # TensorProto.BFLOAT16
}

def _onnx_dtype_to_torch(to_dtype: int):
    """Map ONNX TensorProto dtype integer to torch dtype."""
    dtype_name = _ONNX_TO_TORCH_DTYPE.get(to_dtype, 'float32')
    return getattr(torch, dtype_name)


def _get_attr_ints(node, name, default=None):
    for attr in node.attribute:
        if attr.name == name:
            return list(attr.ints)
    return default


def _get_attr_int(node, name, default=None):
    for attr in node.attribute:
        if attr.name == name:
            return attr.i
    return default


def _get_attr_string(node, name, default=None):
    for attr in node.attribute:
        if attr.name == name:
            s = attr.s
            if isinstance(s, (bytes, bytearray)):
                return s.decode("utf-8")
            return s
    return default


def _onnx_matmul(tensor_values, node, inputs):
    if len(inputs) < 2:
        raise NotImplementedError(
            f"torch reference executor does not yet support op '{node.op_type}'. "
            f"Please add a PyTorch implementation for this op in _execute_onnx_model_torch()."
        )
    a = _np_array_to_torch(inputs[0])
    b = _np_array_to_torch(inputs[1])
    result = torch.matmul(a, b)
    tensor_values[node.output[0]] = _torch_tensor_to_np(result)


def _onnx_einsum(tensor_values, node, inputs):
    eq = _get_attr_string(node, "equation")
    torch_inputs = [_np_array_to_torch(tensor_values[inp]) for inp in node.input]
    result = torch.einsum(eq, *torch_inputs)
    tensor_values[node.output[0]] = _torch_tensor_to_np(result)


def _onnx_maxpool(tensor_values, node, inputs):
    x = _np_array_to_torch(inputs[0])
    kernel_shape = _get_attr_ints(node, "kernel_shape")
    strides = _get_attr_ints(node, "strides")
    if strides is None:
        strides = kernel_shape
    pads = _get_attr_ints(node, "pads")
    if pads is None:
        pads = [0] * (2 * len(kernel_shape))
    ceil_mode = _get_attr_int(node, "ceil_mode", 0)

    ndim = len(kernel_shape)

    # Handle padding manually because ONNX supports asymmetric padding
    # ONNX pads: [d0_begin, d1_begin, ..., dN_begin, d0_end, d1_end, ..., dN_end]
    # PyTorch F.pad for N-D: (last_dim_left, last_dim_right, ..., first_dim_left, first_dim_right)
    if any(p != 0 for p in pads):
        pad_begin = pads[:ndim]
        pad_end = pads[ndim:]
        torch_pad = []
        for i in range(ndim - 1, -1, -1):
            torch_pad.extend([pad_begin[i], pad_end[i]])
        x = F.pad(x, torch_pad, value=-float('inf'))

    pool_kwargs = dict(kernel_size=kernel_shape, stride=strides, padding=0, ceil_mode=bool(ceil_mode))

    if ndim == 1:
        result = F.max_pool1d(x, **pool_kwargs)
    elif ndim == 2:
        result = F.max_pool2d(x, **pool_kwargs)
    elif ndim == 3:
        result = F.max_pool3d(x, **pool_kwargs)
    else:
        raise ValueError(f"Unsupported MaxPool spatial rank: {ndim}")

    tensor_values[node.output[0]] = _torch_tensor_to_np(result)


def _onnx_conv(tensor_values, node, inputs):
    x = _np_array_to_torch(inputs[0])
    w = _np_array_to_torch(inputs[1])
    bias = _np_array_to_torch(inputs[2]) if len(inputs) > 2 else None

    kernel_shape = _get_attr_ints(node, "kernel_shape", [])
    strides = _get_attr_ints(node, "strides", [1] * len(kernel_shape))
    pads = _get_attr_ints(node, "pads", [0] * (2 * len(kernel_shape)))
    dilations = _get_attr_ints(node, "dilations", [1] * len(kernel_shape))
    groups = _get_attr_int(node, "group", 1)

    if len(pads) == 2 * len(kernel_shape):
        is_asymmetric = any(pads[i] != pads[i + len(kernel_shape)] for i in range(len(kernel_shape)))
        if is_asymmetric:
            torch_pad = []
            for i in range(len(kernel_shape) - 1, -1, -1):
                torch_pad.extend([pads[i], pads[i + len(kernel_shape)]])
            x = F.pad(x, torch_pad)
            pads = [0] * len(kernel_shape)
        else:
            pads = [pads[i] for i in range(len(kernel_shape))]

    torch_padding = pads[0] if len(set(pads)) == 1 else tuple(pads)

    if len(kernel_shape) == 1:
        result = F.conv1d(x, w, bias=bias, stride=strides, padding=torch_padding,
                          dilation=dilations, groups=groups)
    elif len(kernel_shape) == 2:
        result = F.conv2d(x, w, bias=bias, stride=strides, padding=torch_padding,
                          dilation=dilations, groups=groups)
    elif len(kernel_shape) == 3:
        result = F.conv3d(x, w, bias=bias, stride=strides, padding=torch_padding,
                          dilation=dilations, groups=groups)
    else:
        raise ValueError(f"Unsupported Conv kernel rank: {len(kernel_shape)}")
    tensor_values[node.output[0]] = _torch_tensor_to_np(result)


def _onnx_cast(tensor_values, node, inputs):
    x = _np_array_to_torch(inputs[0])
    to_dtype = _get_attr_int(node, "to", 1)
    result = x.to(_onnx_dtype_to_torch(to_dtype))
    tensor_values[node.output[0]] = _torch_tensor_to_np(result)


def _onnx_relu(tensor_values, node, inputs):
    x = _np_array_to_torch(inputs[0])
    result = F.relu(x)
    tensor_values[node.output[0]] = _torch_tensor_to_np(result)


def _onnx_concat(tensor_values, node, inputs):
    tensors = [_np_array_to_torch(inp) for inp in inputs]
    axis = _get_attr_int(node, "axis", 0)
    result = torch.cat(tensors, dim=axis)
    tensor_values[node.output[0]] = _torch_tensor_to_np(result)


def _onnx_globalaveragepool(tensor_values, node, inputs):
    x = _np_array_to_torch(inputs[0])
    spatial_rank = x.dim() - 2
    if spatial_rank == 1:
        result = F.adaptive_avg_pool1d(x, 1)
    elif spatial_rank == 2:
        result = F.adaptive_avg_pool2d(x, (1, 1))
    elif spatial_rank == 3:
        result = F.adaptive_avg_pool3d(x, (1, 1, 1))
    else:
        raise ValueError(f"Unsupported GlobalAveragePool spatial rank: {spatial_rank}")
    tensor_values[node.output[0]] = _torch_tensor_to_np(result)


def _onnx_flatten(tensor_values, node, inputs):
    x = inputs[0]
    axis = _get_attr_int(node, "axis", 1)
    # ONNX Flatten: collapse shape[:axis] into one dim, shape[axis:] into one dim
    first_dim = int(np.prod(x.shape[:axis])) if axis > 0 else 1
    shape = [first_dim, -1]
    result = x.reshape(shape)
    tensor_values[node.output[0]] = result


def _onnx_sigmoid(tensor_values, node, inputs):
    x = _np_array_to_torch(inputs[0])
    result = torch.sigmoid(x)
    tensor_values[node.output[0]] = _torch_tensor_to_np(result)


def _onnx_softmax(tensor_values, node, inputs):
    x = _np_array_to_torch(inputs[0])
    axis = _get_attr_int(node, "axis", -1)
    result = F.softmax(x, dim=axis)
    tensor_values[node.output[0]] = _torch_tensor_to_np(result)


def _onnx_add(tensor_values, node, inputs):
    if len(inputs) < 2:
        raise NotImplementedError(
            f"torch reference executor does not yet support op '{node.op_type}'. "
            f"Please add a PyTorch implementation for this op in _execute_onnx_model_torch()."
        )
    a = _np_array_to_torch(inputs[0])
    b = _np_array_to_torch(inputs[1])
    result = torch.add(a, b)
    tensor_values[node.output[0]] = _torch_tensor_to_np(result)


def _onnx_mul(tensor_values, node, inputs):
    if len(inputs) < 2:
        raise NotImplementedError(
            f"torch reference executor does not yet support op '{node.op_type}'. "
            f"Please add a PyTorch implementation for this op in _execute_onnx_model_torch()."
        )
    a = _np_array_to_torch(inputs[0])
    b = _np_array_to_torch(inputs[1])
    result = torch.mul(a, b)
    tensor_values[node.output[0]] = _torch_tensor_to_np(result)


def _onnx_sub(tensor_values, node, inputs):
    if len(inputs) < 2:
        raise NotImplementedError(
            f"torch reference executor does not yet support op '{node.op_type}'. "
            f"Please add a PyTorch implementation for this op in _execute_onnx_model_torch()."
        )
    a = _np_array_to_torch(inputs[0])
    b = _np_array_to_torch(inputs[1])
    result = torch.sub(a, b)
    tensor_values[node.output[0]] = _torch_tensor_to_np(result)


def _onnx_div(tensor_values, node, inputs):
    if len(inputs) < 2:
        raise NotImplementedError(
            f"torch reference executor does not yet support op '{node.op_type}'. "
            f"Please add a PyTorch implementation for this op in _execute_onnx_model_torch()."
        )
    a = _np_array_to_torch(inputs[0])
    b = _np_array_to_torch(inputs[1])
    result = torch.div(a, b)
    tensor_values[node.output[0]] = _torch_tensor_to_np(result)


def _onnx_slice(tensor_values, node, inputs):
    data = _np_array_to_torch(inputs[0])
    starts = inputs[1]
    ends = inputs[2]
    axes = inputs[3] if len(inputs) > 3 else None
    steps = inputs[4] if len(inputs) > 4 else None
    if axes is not None:
        axes = axes.tolist() if hasattr(axes, 'tolist') else list(axes)
    if steps is not None:
        steps = steps.tolist() if hasattr(steps, 'tolist') else list(steps)
    starts = starts.tolist() if hasattr(starts, 'tolist') else list(starts)
    ends = ends.tolist() if hasattr(ends, 'tolist') else list(ends)
    # Build slices for each dimension
    slices = [slice(None)] * data.dim()
    for i, (s, e) in enumerate(zip(starts, ends)):
        axis = axes[i] if axes is not None else i
        step = steps[i] if steps is not None else 1
        slices[axis] = slice(s, e, step)
    result = data[tuple(slices)]
    tensor_values[node.output[0]] = _torch_tensor_to_np(result)


def _onnx_reshape(tensor_values, node, inputs):
    data = _np_array_to_torch(inputs[0])
    shape = inputs[1]
    shape = shape.tolist() if hasattr(shape, 'tolist') else list(shape)
    result = data.reshape(shape)
    tensor_values[node.output[0]] = _torch_tensor_to_np(result)


def _onnx_transpose(tensor_values, node, inputs):
    data = _np_array_to_torch(inputs[0])
    perm = _get_attr_ints(node, "perm", [])
    if not perm:
        perm = list(reversed(range(data.dim())))
    result = data.permute(perm)
    tensor_values[node.output[0]] = _torch_tensor_to_np(result)


def _onnx_gather(tensor_values, node, inputs):
    data = _np_array_to_torch(inputs[0])
    indices = _np_array_to_torch(inputs[1]).long()
    axis = _get_attr_int(node, "axis", 0)
    # Use advanced indexing to match ONNX Gather semantics
    idx = [slice(None)] * data.dim()
    idx[axis] = indices
    result = data[tuple(idx)]
    tensor_values[node.output[0]] = _torch_tensor_to_np(result)


def _onnx_split(tensor_values, node, inputs):
    data = _np_array_to_torch(inputs[0])
    axis = _get_attr_int(node, "axis", 0)
    split_sizes = _get_attr_ints(node, "split", [])
    if not split_sizes and len(inputs) > 1:
        split_sizes = inputs[1].tolist() if hasattr(inputs[1], 'tolist') else list(inputs[1])
    if not split_sizes:
        # Equal split
        num_outputs = len(node.output)
        split_sizes = data.shape[axis] // num_outputs
    result = torch.split(data, split_sizes, dim=axis)
    for i, output_name in enumerate(node.output):
        tensor_values[output_name] = _torch_tensor_to_np(result[i])


def _onnx_unsqueeze(tensor_values, node, inputs):
    data = _np_array_to_torch(inputs[0])
    axes = inputs[1] if len(inputs) > 1 else None
    if axes is not None:
        axes = axes.tolist() if hasattr(axes, 'tolist') else list(axes)
    else:
        axes = _get_attr_ints(node, "axes", [])
    result = data
    for axis in sorted(axes):
        result = result.unsqueeze(axis)
    tensor_values[node.output[0]] = _torch_tensor_to_np(result)


def _onnx_tile(tensor_values, node, inputs):
    data = _np_array_to_torch(inputs[0])
    repeats = inputs[1]
    repeats = repeats.tolist() if hasattr(repeats, 'tolist') else list(repeats)
    result = data.repeat(repeats)
    tensor_values[node.output[0]] = _torch_tensor_to_np(result)


def _onnx_mod(tensor_values, node, inputs):
    a = _np_array_to_torch(inputs[0])
    b = _np_array_to_torch(inputs[1])
    fmod = _get_attr_int(node, "fmod", 0)
    if fmod:
        result = torch.fmod(a, b)
    else:
        result = torch.remainder(a, b)
    tensor_values[node.output[0]] = _torch_tensor_to_np(result)


def _onnx_reducemax(tensor_values, node, inputs):
    data = _np_array_to_torch(inputs[0])
    # axes can be an attribute (older opsets) or an input (opset 18+)
    axes = _get_attr_ints(node, "axes", [])
    if not axes and len(inputs) > 1 and inputs[1] is not None:
        axes_arr = inputs[1]
        axes = axes_arr.tolist() if hasattr(axes_arr, 'tolist') else list(axes_arr)
    keepdims = bool(_get_attr_int(node, "keepdims", 1))
    if not axes:
        result = data.max()
        if keepdims:
            result = result.unsqueeze(0)
    else:
        for axis in sorted(axes, reverse=True):
            result, _ = data.max(dim=axis, keepdim=keepdims)
            data = result
    tensor_values[node.output[0]] = _torch_tensor_to_np(result)


def _onnx_topk(tensor_values, node, inputs):
    data = _np_array_to_torch(inputs[0])
    k = int(inputs[1].item()) if hasattr(inputs[1], 'item') else int(inputs[1])
    axis = _get_attr_int(node, "axis", -1)
    largest = bool(_get_attr_int(node, "largest", 1))
    sorted_attr = bool(_get_attr_int(node, "sorted", 1))
    values, indices = torch.topk(data, k, dim=axis, largest=largest, sorted=sorted_attr)
    tensor_values[node.output[0]] = _torch_tensor_to_np(values)
    tensor_values[node.output[1]] = _torch_tensor_to_np(indices)


def _onnx_resize(tensor_values, node, inputs):
    x = _np_array_to_torch(inputs[0])
    # Read scales or sizes
    sizes = None
    scales = None
    if len(inputs) > 2 and inputs[2] is not None:
        sizes = inputs[2].tolist() if hasattr(inputs[2], 'tolist') else list(inputs[2])
    elif len(inputs) > 1 and inputs[1] is not None:
        scales = inputs[1].tolist() if hasattr(inputs[1], 'tolist') else list(inputs[1])
    mode = _get_attr_string(node, "mode", "nearest")
    if sizes is not None:
        # F.interpolate expects spatial sizes only; drop batch/channel dims for 4D/5D
        if x.dim() >= 3:
            spatial_sizes = sizes[2:]
        else:
            spatial_sizes = sizes
        result = F.interpolate(x, size=spatial_sizes, mode=mode, align_corners=None if mode == "nearest" else False)
    elif scales is not None:
        if x.dim() >= 3 and len(scales) > 2:
            scale_factor = scales[2]
        elif len(scales) > 0:
            scale_factor = scales[-1]
        else:
            scale_factor = None
        result = F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=None if mode == "nearest" else False)
    else:
        raise ValueError("Resize requires scales or sizes")
    tensor_values[node.output[0]] = _torch_tensor_to_np(result)


def _onnx_gatherelements(tensor_values, node, inputs):
    data = _np_array_to_torch(inputs[0])
    indices = _np_array_to_torch(inputs[1])
    axis = _get_attr_int(node, "axis", 0)
    result = torch.gather(data, dim=axis, index=indices)
    tensor_values[node.output[0]] = _torch_tensor_to_np(result)


_ONNX_OP_HANDLERS = {
    'MatMul': _onnx_matmul,
    'Einsum': _onnx_einsum,
    'MaxPool': _onnx_maxpool,
    'Conv': _onnx_conv,
    'Cast': _onnx_cast,
    'Relu': _onnx_relu,
    'Concat': _onnx_concat,
    'GlobalAveragePool': _onnx_globalaveragepool,
    'Flatten': _onnx_flatten,
    'Sigmoid': _onnx_sigmoid,
    'Softmax': _onnx_softmax,
    'Add': _onnx_add,
    'Mul': _onnx_mul,
    'Sub': _onnx_sub,
    'Div': _onnx_div,
    'Slice': _onnx_slice,
    'Reshape': _onnx_reshape,
    'Transpose': _onnx_transpose,
    'Gather': _onnx_gather,
    'Split': _onnx_split,
    'Unsqueeze': _onnx_unsqueeze,
    'Tile': _onnx_tile,
    'Mod': _onnx_mod,
    'ReduceMax': _onnx_reducemax,
    'TopK': _onnx_topk,
    'Resize': _onnx_resize,
    'GatherElements': _onnx_gatherelements,
}


def _execute_onnx_model_torch(model, input_data):
    """
    Execute ONNX model using PyTorch, with special handling for bf16 operations.
    Every op must be explicitly implemented in PyTorch; there is no ONNXRuntime fallback.
    """

    graph = model.graph

    # Build value info map
    value_info_map = {vi.name: vi for vi in list(graph.value_info) + list(graph.input) + list(graph.output)}

    # Initialize tensor values
    tensor_values = {}

    # Load initializers
    for init in graph.initializer:
        arr = numpy_helper.to_array(init)
        tensor_values[init.name] = arr

    # Set input values
    input_names = [inp.name for inp in graph.input]
    for i, inp_name in enumerate(input_names):
        if i < len(input_data):
            tensor_values[inp_name] = input_data[i]

    # Execute nodes in order
    for node in graph.node:
        inputs = [tensor_values.get(inp) for inp in node.input if inp and inp in tensor_values]

        handler = _ONNX_OP_HANDLERS.get(node.op_type)
        if handler is not None:
            handler(tensor_values, node, inputs)
        else:
            raise NotImplementedError(
                f"torch reference executor does not yet support op '{node.op_type}'. "
                f"Please add a PyTorch implementation for this op in _execute_onnx_model_torch()."
            )

    outputs = []
    for output in graph.output:
        if output.name in tensor_values:
            outputs.append(tensor_values[output.name])

    return outputs


@dataclass
class TorchLayerCase(Case):
    """A test case for a layer extracted from a Torch model."""
    layer_name: str
    is_full_model: bool


def _disable_inplace_ops(model):
    """Disable inplace mode on all modules that support it.

    Inplace ops (relu_, hardtanh_, silu_, etc.) cause torch-mlir parsing
    failures with bf16 tensors. This recursively sets inplace=False on
    ReLU, ReLU6, Hardtanh, SiLU, Dropout, etc.
    """
    for module in model.modules():
        if hasattr(module, 'inplace') and module.inplace:
            module.inplace = False
    return model


def _find_layer_by_name(model, layer_name):
    """Find a module by its fully qualified name."""
    for name, module in model.named_modules():
        if name == layer_name:
            return module
    return None


def _extract_tensor_shapes(value):
    """Normalize a tensor, tuple/list of tensors, or dict of tensors to a list
    of shapes. Non-tensor values become ``None``. The returned list preserves
    positional order (dicts are sorted by key to stay deterministic).
    """
    if isinstance(value, torch.Tensor):
        return [tuple(value.shape)]
    if isinstance(value, (list, tuple)):
        return [tuple(v.shape) if isinstance(v, torch.Tensor) else None for v in value]
    if isinstance(value, dict):
        return [
            tuple(v.shape) if isinstance(v, torch.Tensor) else None
            for k, v in sorted(value.items())
        ]
    return [None]


def _load_torch_model_and_inputs(filepath: Path, example_inputs=None):
    """Load a Torch model and normalize example_inputs.

    Supports .pt/.pth bundle dicts and .py files that define a
    ``torch.nn.Module`` subclass with an optional ``get_example_inputs()``.
    """
    suffix = filepath.suffix.lower()
    if suffix in (".pt", ".pth"):
        model, file_example_inputs = _load_model_from_pt(filepath)
    elif suffix == ".py":
        model, file_example_inputs = _load_model_from_py(filepath)
    else:
        raise ValueError(f"Unsupported model file format: {suffix}")

    if example_inputs is None:
        example_inputs = file_example_inputs
    if example_inputs is None:
        raise ValueError(
            f"Model file {filepath} must provide example_inputs "
            f"(either saved in the bundle dict as 'example_inputs', "
            f"or passed directly to generate_torch_layers_from_file())"
        )
    if not isinstance(example_inputs, (list, tuple)):
        example_inputs = (example_inputs,)

    model.eval()
    return model, example_inputs


def _model_file_stat(filepath: Path):
    st = filepath.stat()
    return {"size": st.st_size, "mtime_ns": st.st_mtime_ns}


def _build_torch_layer_cases(filepath: Path, cached_entries):
    """Reconstruct TorchLayerCase objects from cached metadata.

    Cases reference the model by path so that pytest-xdist workers do not
    need to load the model into memory during collection.
    """
    cases = []
    model_path = str(filepath)
    for entry in cached_entries:
        cases.append(TorchLayerCase(
            name=entry["name"],
            data={
                "layer_name": entry["layer_name"],
                "model_path": model_path,
                "is_full_model": entry["is_full_model"],
                "layer_input_shapes": entry.get("layer_input_shapes", []),
                "layer_output_shapes": entry.get("layer_output_shapes", []),
            },
            layer_name=entry["layer_name"],
            is_full_model=entry["is_full_model"],
        ))
    return cases


def _load_torch_layer_cache(key_dir: Path, filepath: Path, node_groups, dedup):
    """Return cached case metadata or None on miss / mismatch / corrupt."""
    manifest_path = key_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
        stat = _model_file_stat(filepath)
        if (manifest.get("version") != _TORCH_LAYER_CACHE_VERSION
                or manifest.get("model_size") != stat["size"]
                or manifest.get("model_mtime_ns") != stat["mtime_ns"]
                or manifest.get("node_groups") != node_groups
                or manifest.get("dedup") != dedup):
            return None
        return _build_torch_layer_cases(filepath, manifest.get("cases", []))
    except Exception as e:
        print(f"[torch-layer-cache] ignoring corrupt cache at {key_dir}: {e}")
        return None


def _save_torch_layer_cache(key_dir: Path, filepath: Path, cases, node_groups, dedup):
    """Atomically write case metadata to the pytest cache directory."""
    stat = _model_file_stat(filepath)
    entries = []
    for c in cases:
        entries.append({
            "name": c.name,
            "layer_name": c.layer_name,
            "is_full_model": c.is_full_model,
            "layer_input_shapes": c.data.get("layer_input_shapes", []),
            "layer_output_shapes": c.data.get("layer_output_shapes", []),
        })
    atomic_write_json_manifest(key_dir, {
        "version": _TORCH_LAYER_CACHE_VERSION,
        "model_size": stat["size"],
        "model_mtime_ns": stat["mtime_ns"],
        "node_groups": node_groups,
        "dedup": dedup,
        "cases": entries,
    })


def generate_torch_layers_from_model(model, example_inputs, node_groups=None, dedup=False):
    """Generate TorchLayerCase objects by extracting layers from a torch.nn.Module.

    For each named module in the model, create a case that exports just that
    submodule to Torch MLIR using the FX importer.

    Args:
        model: A torch.nn.Module or ScriptModule instance.
        example_inputs: Example input tensors for the full model (used to infer
            intermediate shapes for each layer).
        node_groups: Optional list of layer name groups (currently unused for Torch).
        dedup: If True, deduplicate layers with the same class.

    Returns:
        List of TorchLayerCase objects.
    """
    cases = []
    seen = set()

    # ScriptModule doesn't support forward hooks, so we can only test
    # the full model.
    is_scriptmodule = isinstance(model, torch.jit.ScriptModule)

    if is_scriptmodule:
        return [TorchLayerCase(
            name="full_model",
            data={
                "layer_name": "",
                "model": model,
                "is_full_model": True,
                "layer_input_shapes": [tuple(ex.shape) for ex in example_inputs],
                "layer_output_shapes": [],
            },
            layer_name="",
            is_full_model=True,
        )]

    # Run the model once to capture intermediate tensor shapes.
    # trace maps module name -> {"input": [...], "output": [...]}.
    with torch.no_grad():
        trace = {}
        handles = []

        def make_hook(name):
            def hook(mod, inp, out):
                trace[name] = {
                    "input": _extract_tensor_shapes(inp),
                    "output": _extract_tensor_shapes(out),
                }
            return hook

        for name, module in model.named_modules():
            if name:
                handles.append(module.register_forward_hook(make_hook(name)))

        try:
            _ = model(*example_inputs)
        finally:
            for h in handles:
                h.remove()

    for name, module in model.named_modules():
        if not name:
            case_name = "full_model"
            is_full_model = True
            layer_input_shapes = [tuple(ex.shape) for ex in example_inputs]
            layer_output_shapes = []
        else:
            op_type = type(module).__name__
            case_name = f"{name.replace('.', '_')}_{op_type}"
            is_full_model = False
            record = trace.get(name, {})
            layer_input_shapes = record.get("input", [])
            layer_output_shapes = record.get("output", [])

        if dedup:
            key = type(module).__name__
            if key in seen:
                continue
            seen.add(key)

        cases.append(TorchLayerCase(
            name=case_name,
            data={
                "layer_name": name,
                "model": model,
                "is_full_model": is_full_model,
                "layer_input_shapes": layer_input_shapes,
                "layer_output_shapes": layer_output_shapes,
            },
            layer_name=name,
            is_full_model=is_full_model,
        ))

    return cases


def _load_model_from_pt(filepath):
    """Load a model from a .pt/.pth file.

    The file must be a bundle dict saved via:
        torch.save({"model": model, "example_inputs": inputs}, path)

    If a companion .py file exists, it is imported first so pickle can resolve
    the model class.

    Returns (model, example_inputs_or_none).
    """
    py_path = filepath.with_suffix(".py")
    if py_path.exists():
        module_name = py_path.stem
        spec = importlib.util.spec_from_file_location(module_name, py_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

    checkpoint = torch.load(filepath, weights_only=False)
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(
            f"Model file {filepath} must be a dict with 'model' and 'example_inputs' keys, "
            f"saved via torch.save({{'model': model, 'example_inputs': inputs}}, path)"
        )
    model = checkpoint["model"]
    if not isinstance(model, torch.nn.Module):
        raise ValueError(f"'model' must be a torch.nn.Module, got {type(model)}")
    return model, checkpoint.get("example_inputs")


def _find_model_class_in_module(module, preferred_name=None):
    """Find a torch.nn.Module subclass defined in the module.

    If ``preferred_name`` is given and a class with that exact name exists and
    is a Module subclass, return it. Otherwise return the first Module subclass
    found in the module namespace. This makes .py model loading deterministic
    when the file name matches the model class name.
    """
    candidates = []
    for name, obj in vars(module).items():
        if (isinstance(obj, type) and
            issubclass(obj, torch.nn.Module) and
            obj is not torch.nn.Module and
            obj.__module__ == module.__name__):
            if preferred_name and name == preferred_name:
                return obj
            candidates.append(obj)
    return candidates[0] if candidates else None


def _load_model_from_py(filepath):
    """Load a model from a .py file.

    Finds the first torch.nn.Module subclass defined in the file,
    instantiates it, and returns it.

    Returns (model, example_inputs_or_none).
    """
    spec = importlib.util.spec_from_file_location("model_module", filepath)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    model_cls = _find_model_class_in_module(module, preferred_name=filepath.stem)
    if model_cls is None:
        raise ValueError(f"Model file {filepath} must define a torch.nn.Module subclass")

    model = model_cls()
    model.eval()
    file_example_inputs = module.get_example_inputs() if hasattr(module, "get_example_inputs") else None
    return model, file_example_inputs


def generate_torch_layers_from_file(filepath, example_inputs=None, node_groups=None, dedup=False, cache=None):
    """Load a model from a file and generate layer cases.

    Supports:
      - Python files (.py) that define a torch.nn.Module subclass and optionally
        expose ``get_example_inputs()``.
      - PyTorch files (.pt, .pth) containing a bundle dict:
        ``torch.save({"model": model, "example_inputs": inputs}, path)``.

    When ``cache`` is provided (pytest's ``config.cache``), the extracted layer
    metadata is cached on disk and shared across pytest-xdist workers. The first
    worker performs the potentially expensive forward-hook trace; subsequent
    workers load the cached metadata and the model file, avoiding redundant
    extraction. This mirrors the caching strategy used by the ONNX and TFLite
    layer generators.
    """
    filepath = Path(filepath)

    # Caching is only safe when we rely on the file's own example_inputs, so
    # that the cache key is determined by the model file's mtime/size.
    if cache is not None and example_inputs is None:
        cache_dir = cache.mkdir('torch_layer_cache')
        key_dir = cache_dir / filepath.stem
        # Keep the lock file outside key_dir so the directory can be replaced
        # safely under pytest-xdist / NFS.
        lock_path = cache_dir / (filepath.stem + ".lock")
        key_dir.mkdir(parents=True, exist_ok=True)

        with FileLock(str(lock_path)):
            cached_cases = _load_torch_layer_cache(key_dir, filepath, node_groups, dedup)
            if cached_cases is not None:
                print(f"[torch-layer-cache] HIT  {filepath.stem} -> {key_dir}")
                return cached_cases

            print(f"[torch-layer-cache] MISS {filepath.stem} -> {key_dir}")
            model, example_inputs = _load_torch_model_and_inputs(filepath, None)
            cases = generate_torch_layers_from_model(model, example_inputs, node_groups, dedup)
            # Replace the in-memory model object with a path reference so that
            # pytest-xdist workers do not all keep the full model in memory.
            model_path = str(filepath)
            for c in cases:
                c.data["model_path"] = model_path
                c.data.pop("model", None)
            _save_torch_layer_cache(key_dir, filepath, cases, node_groups, dedup)
            return cases

    model, example_inputs = _load_torch_model_and_inputs(filepath, example_inputs)
    return generate_torch_layers_from_model(model, example_inputs, node_groups, dedup)


def _save_torch_metadata_cache(key_dir: Path, model_version: str, cases, node_groups, dedup):
    """Atomically write lightweight case metadata (no model weights) to disk."""
    entries = []
    for c in cases:
        entries.append({
            "name": c.name,
            "layer_name": c.layer_name,
            "is_full_model": c.is_full_model,
            "layer_input_shapes": c.data.get("layer_input_shapes", []),
            "layer_output_shapes": c.data.get("layer_output_shapes", []),
        })
    atomic_write_json_manifest(key_dir, {
        "version": _TORCH_LAYER_CACHE_VERSION,
        "model_version": model_version,
        "node_groups": node_groups,
        "dedup": dedup,
        "cases": entries,
    })


def _load_torch_metadata_cases(manifest: dict, model_loader_key: str):
    """Build TorchLayerCase objects from a metadata-only manifest."""
    cases = []
    for entry in manifest.get("cases", []):
        cases.append(TorchLayerCase(
            name=entry["name"],
            data={
                "layer_name": entry["layer_name"],
                "model_loader_key": model_loader_key,
                "is_full_model": entry["is_full_model"],
                "layer_input_shapes": entry.get("layer_input_shapes", []),
                "layer_output_shapes": entry.get("layer_output_shapes", []),
            },
            layer_name=entry["layer_name"],
            is_full_model=entry["is_full_model"],
        ))
    return cases


def generate_torch_layers_from_model_metadata_cache(
    cache,
    prefix: str,
    model_version: str,
    model_loader_key: str,
    model_fn,
    example_inputs,
    node_groups=None,
    dedup=False,
):
    """Generate layer cases for an in-memory model, caching only metadata.

    This is designed for models that are expensive to instantiate (e.g.
    torchvision zoo models).  The first pytest-xdist worker that reaches this
    function builds the model, runs the forward-hook trace, and writes a small
    metadata manifest.  Subsequent workers load only the manifest, so they do
    not need to download weights or load the model during collection.

    The returned cases reference the model via ``model_loader_key``; fixtures
    load the actual model lazily and cache it per worker process.
    """
    cache_dir = cache.mkdir('torch_model_metadata_cache')
    key_dir = cache_dir / prefix
    # Keep the lock file outside key_dir so the directory can be replaced
    # safely under pytest-xdist / NFS.
    lock_path = cache_dir / (prefix + ".lock")
    key_dir.mkdir(parents=True, exist_ok=True)

    with FileLock(str(lock_path)):
        manifest_path = key_dir / "manifest.json"
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                if (manifest.get("version") == _TORCH_LAYER_CACHE_VERSION
                        and manifest.get("model_version") == model_version
                        and manifest.get("node_groups") == node_groups
                        and manifest.get("dedup") == dedup):
                    print(f"[torch-metadata-cache] HIT  {prefix} -> {key_dir}")
                    return _load_torch_metadata_cases(manifest, model_loader_key)
            except Exception as e:
                print(f"[torch-metadata-cache] ignoring corrupt cache for {prefix}: {e}")

        print(f"[torch-metadata-cache] MISS {prefix} -> {key_dir}")
        model = model_fn()
        if not isinstance(model, torch.nn.Module):
            raise ValueError(f"model_fn for {prefix} must return a torch.nn.Module, got {type(model)}")
        if not isinstance(example_inputs, (list, tuple)):
            example_inputs = (example_inputs,)
        model.eval()
        cases = generate_torch_layers_from_model(model, example_inputs, node_groups, dedup)
        for c in cases:
            c.name = f"{prefix}_{c.name}"
            c.data["model_loader_key"] = model_loader_key
        _save_torch_metadata_cache(key_dir, model_version, cases, node_groups, dedup)
        return cases


@pytest.fixture
def torch_layer_model_data(request, case_config):
    """Fixture that exports a single layer from a Torch model to Torch MLIR text.

    Returns a VersionedUncachedData containing the MLIR text.
    The actual file generation and caching is handled by torch_layer_model.
    """
    from iree.compiler.extras.fx_importer import FxImporter

    layer_name = case_config["layer_name"]
    model = _get_model_from_case_config(case_config)
    layer_input_shapes = case_config.get("layer_input_shapes", [])

    # Find the submodule
    submodule = _find_layer_by_name(model, layer_name) if layer_name else model
    if submodule is None:
        raise ValueError(f"Layer '{layer_name}' not found in model")

    # Build example inputs from recorded shapes (random data is fine for export)
    # Use bfloat16 to match the target hardware precision.
    example_inputs = []
    for shape in layer_input_shapes:
        if shape is None:
            continue
        example_inputs.append(torch.randn(*shape, dtype=torch.bfloat16))

    if not example_inputs:
        raise ValueError(f"No input shapes available for layer '{layer_name}'")

    exported = torch.export.export(submodule, tuple(example_inputs))

    # Import to Torch MLIR
    from iree.compiler import ir
    ctx = ir.Context()
    importer = FxImporter(context=ctx)
    importer.import_frozen_program(exported)
    mlir_module = importer.module

    mlir_text = str(mlir_module)
    version = f"torch_layer_{layer_name}_{type(model).__name__}"
    return VersionedUncachedData(data=mlir_text, version=version)


@pytest.fixture
def torch_model_data(request, case_config):
    return request.getfixturevalue(case_config.get('torch_model_data', 'torch_layer_model_data'))


@versioned_generated_file_fixture("mlir")
def torch_layer_model(request, versioned_file, torch_model_data):
    """Fixture that writes the Torch MLIR text to a versioned cached file."""
    with open(versioned_file, "w") as f:
        f.write(torch_model_data)


@pytest.fixture
def torch_mlir_model_file(request, case_config):
    """Alternative fixture name for torch_layer_model."""
    return request.getfixturevalue("torch_layer_model")


def _execute_torch_module_direct(case_config, input_data):
    """Run the original PyTorch model or a specific layer directly.

    Args:
        case_config: Dictionary with 'model' and 'layer_name' keys.
        input_data: Numpy array(s) to feed into the model.

    Returns:
        List of numpy arrays, or None if case_config doesn't have model info.
    """
    if "layer_name" not in case_config:
        return None

    model = _get_model_from_case_config(case_config)
    layer_name = case_config["layer_name"]

    submodule = _find_layer_by_name(model, layer_name) if layer_name else model
    if submodule is None:
        return None

    # Determine the model's dtype from its parameters.
    # Layers like ReLU have no parameters, so fall back to the full model.
    try:
        model_dtype = next(submodule.parameters()).dtype
    except StopIteration:
        model_dtype = next(model.parameters()).dtype

    # Convert numpy input data to torch tensors with matching dtype.
    # ml_dtypes.bfloat16 is not supported by torch.from_numpy, so
    # convert via float32 first.
    def _to_torch(np_arr):
        if hasattr(np_arr, 'dtype') and str(np_arr.dtype) == 'bfloat16':
            return torch.tensor(np_arr.astype(np.float32)).to(model_dtype)
        return torch.from_numpy(np_arr).to(model_dtype)

    if isinstance(input_data, (list, tuple)):
        torch_inputs = [_to_torch(x) for x in input_data]
    else:
        torch_inputs = [_to_torch(input_data)]

    # The FX importer lifts model parameters (weights, BN stats) into function
    # arguments, so mlir_io_spec may report more "inputs" than the actual
    # submodule takes. Use layer_input_shapes to determine the real input count.
    # The actual user inputs are the LAST arguments in the MLIR function.
    layer_input_shapes = case_config.get("layer_input_shapes", [])
    real_input_count = len([s for s in layer_input_shapes if s is not None])
    if real_input_count > 0 and len(torch_inputs) > real_input_count:
        torch_inputs = torch_inputs[-real_input_count:]

    with torch.no_grad():
        outputs = submodule(*torch_inputs)

    if isinstance(outputs, torch.Tensor):
        outputs = [outputs]
    else:
        outputs = list(outputs) if isinstance(outputs, (list, tuple)) else [outputs]

    # Convert to numpy. bf16 is not supported by numpy, so cast to f32 first.
    return [x.detach().cpu().to(torch.float32).numpy() for x in outputs]


@versioned_unhashable_object_fixture
def torch_reference_results(request, input_data):
    """Generate reference using PyTorch for ops that numpy/ONNXRuntime can't handle.

    For ONNX-based tests, an onnx_model_file fixture is used directly.
    For torch model tests, the original PyTorch model/layer is run directly.
    For torch-mlir tests without an ONNX model, the torch MLIR file is parsed
    to construct a minimal ONNX model on the fly.
    """
    try:
        onnx_model_file = request.getfixturevalue("onnx_model_file")
        onnx_path = onnx_model_file.file_path if hasattr(onnx_model_file, 'file_path') else Path(str(onnx_model_file))
        onnx_model = onnx.load(str(onnx_path))
        return _execute_onnx_model_torch(onnx_model, input_data)
    except Exception:
        pass

    # Try to run the original PyTorch model/layer directly.
    try:
        case_config = request.getfixturevalue("case_config")
        result = _execute_torch_module_direct(case_config, input_data)
        if result is not None:
            return result
    except Exception:
        pass

    mlir_model_file = request.getfixturevalue("mlir_model_file")
    mlir_path = mlir_model_file.file_path if hasattr(mlir_model_file, 'file_path') else Path(str(mlir_model_file))

    return _execute_torch_mlir(mlir_path, input_data)
