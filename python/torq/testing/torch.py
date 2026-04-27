import numpy as np
import onnx
import onnxruntime
from onnx import helper, numpy_helper, TensorProto

from .versioned_fixtures import versioned_unhashable_object_fixture


def _np_array_to_torch(arr):
    """Convert numpy array to torch tensor, handling bfloat16 dtype."""
    import torch
    if str(arr.dtype) == 'bfloat16':
        return torch.from_numpy(arr.view(np.uint16)).view(torch.bfloat16)
    return torch.from_numpy(arr)


def _torch_tensor_to_np(tensor):
    """Convert torch tensor to numpy array, handling bfloat16 dtype."""
    import torch
    if tensor.dtype == torch.bfloat16:
        arr = tensor.view(torch.uint16).numpy()
        try:
            return arr.view('bfloat16')
        except TypeError:
            import ml_dtypes
            return arr.view(ml_dtypes.bfloat16)
    return tensor.numpy()


def _execute_onnx_model_torch(model, input_data):
    """
    Execute ONNX model using PyTorch, with special handling for bf16 operations.
    Uses torch for Conv/Pool/MatMul/Einsum/Activations and onnxruntime as fallback
    for other operations.
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
            import torch
            a = _np_array_to_torch(inputs[0])
            b = _np_array_to_torch(inputs[1])
            result = torch.matmul(a, b)
            tensor_values[node.output[0]] = _torch_tensor_to_np(result)

        elif node.op_type == "Einsum":
            import torch
            eq = next((attr.s.decode("utf-8") if isinstance(attr.s, (bytes, bytearray)) else attr.s)
                    for attr in node.attribute if attr.name == "equation")
            torch_inputs = [_np_array_to_torch(tensor_values[inp]) for inp in node.input]
            result = torch.einsum(eq, *torch_inputs)
            tensor_values[node.output[0]] = _torch_tensor_to_np(result)

        elif node.op_type == "MaxPool":
            import torch.nn.functional as F
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
            import torch.nn.functional as F
            x = _np_array_to_torch(inputs[0])
            w = _np_array_to_torch(inputs[1])
            bias = _np_array_to_torch(inputs[2]) if len(inputs) > 2 else None

            kernel_shape = list(next((attr.ints for attr in node.attribute if attr.name == "kernel_shape"), []))
            strides = list(next((attr.ints for attr in node.attribute if attr.name == "strides"), [1] * len(kernel_shape)))
            pads = list(next((attr.ints for attr in node.attribute if attr.name == "pads"), [0] * (2 * len(kernel_shape))))
            dilations = list(next((attr.ints for attr in node.attribute if attr.name == "dilations"), [1] * len(kernel_shape)))
            groups = next((attr.i for attr in node.attribute if attr.name == "groups"), 1)

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

        elif node.op_type == "Relu":
            import torch.nn.functional as F
            x = _np_array_to_torch(inputs[0])
            result = F.relu(x)
            tensor_values[node.output[0]] = _torch_tensor_to_np(result)

        elif node.op_type == "Concat":
            import torch
            tensors = [_np_array_to_torch(inp) for inp in inputs]
            axis = next((attr.i for attr in node.attribute if attr.name == "axis"), 0)
            result = torch.cat(tensors, dim=axis)
            tensor_values[node.output[0]] = _torch_tensor_to_np(result)

        elif node.op_type == "GlobalAveragePool":
            import torch.nn.functional as F
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
            shape = list(x.shape[:axis]) + [-1]
            result = x.reshape(shape)
            tensor_values[node.output[0]] = result

        else:
            try:
                node_inputs = [helper.make_tensor_value_info(inp, TensorProto.FLOAT, [])
                               for inp in node.input if inp in tensor_values]
                node_outputs = [helper.make_tensor_value_info(out, TensorProto.FLOAT, [])
                                for out in node.output]
                node_inits = [init for init in graph.initializer if init.name in node.input]

                temp_graph = helper.make_graph([node], 'temp_graph', node_inputs, node_outputs, node_inits)
                temp_model = helper.make_model(temp_graph)
                temp_model.opset_import.extend(model.opset_import)
                temp_model.ir_version = model.ir_version
                ort_session = onnxruntime.InferenceSession(temp_model.SerializeToString())
                ort_inputs = {inp: tensor_values[inp] for inp in node.input if inp in tensor_values}
                outputs = ort_session.run(None, ort_inputs)
                for i, output_name in enumerate(node.output):
                    tensor_values[output_name] = outputs[i]
            except Exception as e:
                print(f"Warning: Could not execute {node.op_type} with onnxruntime: {e}")
                raise

    outputs = []
    for output in graph.output:
        if output.name in tensor_values:
            outputs.append(tensor_values[output.name])

    return outputs


@versioned_unhashable_object_fixture
def torch_reference_results(request, onnx_model_file, input_data):
    """Generate reference using PyTorch for ops that numpy/ONNXRuntime can't handle."""
    onnx_model = onnx.load(str(onnx_model_file))
    return _execute_onnx_model_torch(onnx_model, input_data)
