import numpy as np
import onnx
import re
from pathlib import Path
from onnx import helper, numpy_helper, TensorProto
try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None

from .versioned_fixtures import versioned_unhashable_object_fixture


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

        if node.op_type == 'MatMul' and len(inputs) >= 2:
            a = _np_array_to_torch(inputs[0])
            b = _np_array_to_torch(inputs[1])
            result = torch.matmul(a, b)
            tensor_values[node.output[0]] = _torch_tensor_to_np(result)

        elif node.op_type == "Einsum":
            eq = next((attr.s.decode("utf-8") if isinstance(attr.s, (bytes, bytearray)) else attr.s)
                    for attr in node.attribute if attr.name == "equation")
            torch_inputs = [_np_array_to_torch(tensor_values[inp]) for inp in node.input]
            result = torch.einsum(eq, *torch_inputs)
            tensor_values[node.output[0]] = _torch_tensor_to_np(result)

        elif node.op_type == "MaxPool":
            x = _np_array_to_torch(inputs[0])
            kernel_shape = list(next(attr.ints for attr in node.attribute if attr.name == "kernel_shape"))
            strides = list(next(attr.ints for attr in node.attribute if attr.name == "strides")) if any(attr.name == "strides" for attr in node.attribute) else kernel_shape
            pads = list(next(attr.ints for attr in node.attribute if attr.name == "pads")) if any(attr.name == "pads" for attr in node.attribute) else [0] * (2 * len(kernel_shape))
            ceil_mode = next((attr.i for attr in node.attribute if attr.name == "ceil_mode"), 0)

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

        elif node.op_type == "Conv":
            x = _np_array_to_torch(inputs[0])
            w = _np_array_to_torch(inputs[1])
            bias = _np_array_to_torch(inputs[2]) if len(inputs) > 2 else None

            kernel_shape = list(next((attr.ints for attr in node.attribute if attr.name == "kernel_shape"), []))
            strides = list(next((attr.ints for attr in node.attribute if attr.name == "strides"), [1] * len(kernel_shape)))
            pads = list(next((attr.ints for attr in node.attribute if attr.name == "pads"), [0] * (2 * len(kernel_shape))))
            dilations = list(next((attr.ints for attr in node.attribute if attr.name == "dilations"), [1] * len(kernel_shape)))
            groups = next((attr.i for attr in node.attribute if attr.name == "group"), 1)

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

        elif node.op_type == "Cast":
            x = _np_array_to_torch(inputs[0])
            to_dtype = next((attr.i for attr in node.attribute if attr.name == "to"), 1)
            result = x.to(_onnx_dtype_to_torch(to_dtype))
            tensor_values[node.output[0]] = _torch_tensor_to_np(result)

        elif node.op_type == "Relu":
            x = _np_array_to_torch(inputs[0])
            result = F.relu(x)
            tensor_values[node.output[0]] = _torch_tensor_to_np(result)

        elif node.op_type == "Concat":
            tensors = [_np_array_to_torch(inp) for inp in inputs]
            axis = next((attr.i for attr in node.attribute if attr.name == "axis"), 0)
            result = torch.cat(tensors, dim=axis)
            tensor_values[node.output[0]] = _torch_tensor_to_np(result)

        elif node.op_type == "GlobalAveragePool":
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

        elif node.op_type == "Flatten":
            x = inputs[0]
            axis = next((attr.i for attr in node.attribute if attr.name == "axis"), 1)
            # ONNX Flatten: collapse shape[:axis] into one dim, shape[axis:] into one dim
            import numpy as np
            first_dim = int(np.prod(x.shape[:axis])) if axis > 0 else 1
            shape = [first_dim, -1]
            result = x.reshape(shape)
            tensor_values[node.output[0]] = result

        elif node.op_type == "Sigmoid":
            x = _np_array_to_torch(inputs[0])
            result = torch.sigmoid(x)
            tensor_values[node.output[0]] = _torch_tensor_to_np(result)

        elif node.op_type == "Softmax":
            x = _np_array_to_torch(inputs[0])
            axis = next((attr.i for attr in node.attribute if attr.name == "axis"), -1)
            result = F.softmax(x, dim=axis)
            tensor_values[node.output[0]] = _torch_tensor_to_np(result)

        elif node.op_type == "Add" and len(inputs) >= 2:
            a = _np_array_to_torch(inputs[0])
            b = _np_array_to_torch(inputs[1])
            result = torch.add(a, b)
            tensor_values[node.output[0]] = _torch_tensor_to_np(result)

        elif node.op_type == "Mul" and len(inputs) >= 2:
            a = _np_array_to_torch(inputs[0])
            b = _np_array_to_torch(inputs[1])
            result = torch.mul(a, b)
            tensor_values[node.output[0]] = _torch_tensor_to_np(result)

        elif node.op_type == "Sub" and len(inputs) >= 2:
            a = _np_array_to_torch(inputs[0])
            b = _np_array_to_torch(inputs[1])
            result = torch.sub(a, b)
            tensor_values[node.output[0]] = _torch_tensor_to_np(result)

        elif node.op_type == "Div" and len(inputs) >= 2:
            a = _np_array_to_torch(inputs[0])
            b = _np_array_to_torch(inputs[1])
            result = torch.div(a, b)
            tensor_values[node.output[0]] = _torch_tensor_to_np(result)

        elif node.op_type == "Slice":
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

        elif node.op_type == "Reshape":
            data = _np_array_to_torch(inputs[0])
            shape = inputs[1]
            shape = shape.tolist() if hasattr(shape, 'tolist') else list(shape)
            result = data.reshape(shape)
            tensor_values[node.output[0]] = _torch_tensor_to_np(result)

        elif node.op_type == "Transpose":
            data = _np_array_to_torch(inputs[0])
            perm = list(next((attr.ints for attr in node.attribute if attr.name == "perm"), []))
            if not perm:
                perm = list(reversed(range(data.dim())))
            result = data.permute(perm)
            tensor_values[node.output[0]] = _torch_tensor_to_np(result)

        elif node.op_type == "Gather":
            data = _np_array_to_torch(inputs[0])
            indices = _np_array_to_torch(inputs[1]).long()
            axis = next((attr.i for attr in node.attribute if attr.name == "axis"), 0)
            # Use advanced indexing to match ONNX Gather semantics
            idx = [slice(None)] * data.dim()
            idx[axis] = indices
            result = data[tuple(idx)]
            tensor_values[node.output[0]] = _torch_tensor_to_np(result)

        elif node.op_type == "Split":
            data = _np_array_to_torch(inputs[0])
            axis = next((attr.i for attr in node.attribute if attr.name == "axis"), 0)
            split_sizes = list(next((attr.ints for attr in node.attribute if attr.name == "split"), []))
            if not split_sizes and len(inputs) > 1:
                split_sizes = inputs[1].tolist() if hasattr(inputs[1], 'tolist') else list(inputs[1])
            if not split_sizes:
                # Equal split
                num_outputs = len(node.output)
                split_sizes = data.shape[axis] // num_outputs
            result = torch.split(data, split_sizes, dim=axis)
            for i, output_name in enumerate(node.output):
                tensor_values[output_name] = _torch_tensor_to_np(result[i])

        elif node.op_type == "Unsqueeze":
            data = _np_array_to_torch(inputs[0])
            axes = inputs[1] if len(inputs) > 1 else None
            if axes is not None:
                axes = axes.tolist() if hasattr(axes, 'tolist') else list(axes)
            else:
                axes = list(next((attr.ints for attr in node.attribute if attr.name == "axes"), []))
            result = data
            for axis in sorted(axes):
                result = result.unsqueeze(axis)
            tensor_values[node.output[0]] = _torch_tensor_to_np(result)

        elif node.op_type == "Tile":
            data = _np_array_to_torch(inputs[0])
            repeats = inputs[1]
            repeats = repeats.tolist() if hasattr(repeats, 'tolist') else list(repeats)
            result = data.repeat(repeats)
            tensor_values[node.output[0]] = _torch_tensor_to_np(result)

        elif node.op_type == "Mod":
            a = _np_array_to_torch(inputs[0])
            b = _np_array_to_torch(inputs[1])
            fmod = next((attr.i for attr in node.attribute if attr.name == "fmod"), 0)
            if fmod:
                result = torch.fmod(a, b)
            else:
                result = torch.remainder(a, b)
            tensor_values[node.output[0]] = _torch_tensor_to_np(result)

        elif node.op_type == "ReduceMax":
            data = _np_array_to_torch(inputs[0])
            # axes can be an attribute (older opsets) or an input (opset 18+)
            axes = list(next((attr.ints for attr in node.attribute if attr.name == "axes"), []))
            if not axes and len(inputs) > 1 and inputs[1] is not None:
                axes_arr = inputs[1]
                axes = axes_arr.tolist() if hasattr(axes_arr, 'tolist') else list(axes_arr)
            keepdims = bool(next((attr.i for attr in node.attribute if attr.name == "keepdims"), 1))
            if not axes:
                result = data.max()
                if keepdims:
                    result = result.unsqueeze(0)
            else:
                for axis in sorted(axes, reverse=True):
                    result, _ = data.max(dim=axis, keepdim=keepdims)
                    data = result
            tensor_values[node.output[0]] = _torch_tensor_to_np(result)

        elif node.op_type == "TopK":
            data = _np_array_to_torch(inputs[0])
            k = int(inputs[1].item()) if hasattr(inputs[1], 'item') else int(inputs[1])
            axis = next((attr.i for attr in node.attribute if attr.name == "axis"), -1)
            largest = bool(next((attr.i for attr in node.attribute if attr.name == "largest"), 1))
            sorted_attr = bool(next((attr.i for attr in node.attribute if attr.name == "sorted"), 1))
            values, indices = torch.topk(data, k, dim=axis, largest=largest, sorted=sorted_attr)
            tensor_values[node.output[0]] = _torch_tensor_to_np(values)
            tensor_values[node.output[1]] = _torch_tensor_to_np(indices)

        elif node.op_type == "Resize":
            x = _np_array_to_torch(inputs[0])
            # Read scales or sizes
            sizes = None
            scales = None
            if len(inputs) > 2 and inputs[2] is not None:
                sizes = inputs[2].tolist() if hasattr(inputs[2], 'tolist') else list(inputs[2])
            elif len(inputs) > 1 and inputs[1] is not None:
                scales = inputs[1].tolist() if hasattr(inputs[1], 'tolist') else list(inputs[1])
            mode = next((attr.s.decode("utf-8") if isinstance(attr.s, (bytes, bytearray)) else attr.s
                        for attr in node.attribute if attr.name == "mode"), "nearest")
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

        elif node.op_type == "GatherElements":
            data = _np_array_to_torch(inputs[0])
            indices = _np_array_to_torch(inputs[1])
            axis = next((attr.i for attr in node.attribute if attr.name == "axis"), 0)
            result = torch.gather(data, dim=axis, index=indices)
            tensor_values[node.output[0]] = _torch_tensor_to_np(result)

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


@versioned_unhashable_object_fixture
def torch_reference_results(request, input_data):
    """Generate reference using PyTorch for ops that numpy/ONNXRuntime can't handle.

    For ONNX-based tests, an onnx_model_file fixture is used directly.
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

    # Fallback: execute torch MLIR directly with PyTorch
    mlir_model_file = request.getfixturevalue("mlir_model_file")
    # mlir_model_file is a VersionedFile with .file_path attribute
    mlir_path = mlir_model_file.file_path if hasattr(mlir_model_file, 'file_path') else Path(str(mlir_model_file))
    return _execute_torch_mlir(mlir_path, input_data)
