import numpy as np
import onnx
import onnxruntime
from onnx import helper, numpy_helper, TensorProto

from .versioned_fixtures import versioned_unhashable_object_fixture


def _has_bf16_matmul(model):
    """Check if model contains MatMul with bf16."""
    graph = model.graph
    value_info = {vi.name: vi.type.tensor_type.elem_type
                  for vi in list(graph.value_info) + list(graph.input) + list(graph.output)
                  if hasattr(vi.type, 'tensor_type')}
    for node in graph.node:
        if node.op_type == 'MatMul':
            node_inputs = list(node.input)
            node_outputs = list(node.output)
            if any(value_info.get(name) == TensorProto.BFLOAT16 for name in node_inputs + node_outputs):
                return True
            if any(init.data_type == TensorProto.BFLOAT16 for init in graph.initializer
                   if init.name in node_inputs):
                return True
    return False


def _has_bf16_einsum(model):
    """Check if model contains Einsum with bf16."""
    graph = model.graph
    value_info = {vi.name: vi.type.tensor_type.elem_type
                  for vi in list(graph.value_info) + list(graph.input) + list(graph.output)
                  if hasattr(vi.type, 'tensor_type')}
    for node in graph.node:
        if node.op_type == 'Einsum':
            node_inputs = list(node.input)
            node_outputs = list(node.output)
            if any(value_info.get(name) == TensorProto.BFLOAT16 for name in node_inputs + node_outputs):
                return True
            if any(init.data_type == TensorProto.BFLOAT16 for init in graph.initializer
                   if init.name in node_inputs):
                return True
    return False


def _numpy_maxpool(x, kernel_shape, strides, pads, ceil_mode=0):
    """Execute MaxPool using numpy (NCHW format)."""
    N, C, H, W = x.shape
    kh, kw = kernel_shape
    sh, sw = strides
    pad_top, pad_left, pad_bottom, pad_right = pads

    # Pad the input with -inf for max pooling
    x_padded = np.pad(
        x, ((0, 0), (0, 0), (pad_top, pad_bottom), (pad_left, pad_right)),
        mode='constant', constant_values=-np.inf
    )

    # Compute output dimensions
    padded_h = H + pad_top + pad_bottom
    padded_w = W + pad_left + pad_right
    if ceil_mode:
        H_out = int(np.ceil((padded_h - kh) / sh)) + 1
        W_out = int(np.ceil((padded_w - kw) / sw)) + 1
    else:
        H_out = int(np.floor((padded_h - kh) / sh)) + 1
        W_out = int(np.floor((padded_w - kw) / sw)) + 1

    # Use as_strided for efficient sliding window
    itemsize = x_padded.dtype.itemsize
    windows = np.lib.stride_tricks.as_strided(
        x_padded,
        shape=(N, C, H_out, W_out, kh, kw),
        strides=(
            x_padded.strides[0], x_padded.strides[1],
            x_padded.strides[2] * sh, x_padded.strides[3] * sw,
            x_padded.strides[2], x_padded.strides[3]
        ),
        writeable=False
    )
    return np.max(windows, axis=(4, 5))


def _execute_onnx_model_numpy(model, input_data):
    """
    Execute ONNX model using numpy, with special handling for bf16 MatMul operations.
    NumPy's matmul accepts bf16 arrays directly and handles promotion to float32 internally,
    then we convert the result back to bf16 if needed. Uses manual execution with numpy
    for MatMul and onnxruntime for other operations.
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
            result = np.matmul(inputs[0], inputs[1])
            # Convert to bf16 if output type requires it
            if node.output and node.output[0] in value_info_map:
                vi = value_info_map[node.output[0]]
                if hasattr(vi.type, 'tensor_type') and vi.type.tensor_type.elem_type == TensorProto.BFLOAT16:
                    result = result.astype('bfloat16')
            tensor_values[node.output[0]] = result

        elif node.op_type == "Einsum":
            # Get equation attribute
            eq = next((attr.s.decode("utf-8") if isinstance(attr.s, (bytes, bytearray)) else attr.s)
                    for attr in node.attribute if attr.name == "equation")

            # Gather inputs in order
            a = tensor_values[node.input[0]]
            b = tensor_values[node.input[1]]

            if eq == "n,d->nd":
                result = a[:, None] * b[None, :]
            else:
                result = np.einsum(eq, a, b)
            tensor_values[node.output[0]] = result

        elif node.op_type == "MaxPool":
            x = tensor_values[node.input[0]]
            kernel_shape = list(next(attr.ints for attr in node.attribute if attr.name == "kernel_shape"))
            strides = list(next(attr.ints for attr in node.attribute if attr.name == "strides")) if any(attr.name == "strides" for attr in node.attribute) else kernel_shape
            pads = list(next(attr.ints for attr in node.attribute if attr.name == "pads")) if any(attr.name == "pads" for attr in node.attribute) else [0, 0, 0, 0]
            ceil_mode = next((attr.i for attr in node.attribute if attr.name == "ceil_mode"), 0)
            result = _numpy_maxpool(x, kernel_shape, strides, pads, ceil_mode)
            tensor_values[node.output[0]] = result

        else:
            # For other operations, try to use onnxruntime
            try:
                # Create a temporary model with just this node and required inputs/outputs
                node_inputs = [helper.make_tensor_value_info(inp, TensorProto.FLOAT, [])
                               for inp in node.input if inp in tensor_values]
                node_outputs = [helper.make_tensor_value_info(out, TensorProto.FLOAT, [])
                                for out in node.output]
                node_inits = [init for init in graph.initializer if init.name in node.input]

                temp_graph = helper.make_graph([node], 'temp_graph', node_inputs, node_outputs, node_inits)
                temp_model = helper.make_model(temp_graph)
                # Copy opset from original model so ONNXRuntime doesn't complain
                # about unsupported default opset version
                temp_model.opset_import.extend(model.opset_import)
                temp_model.ir_version = model.ir_version
                ort_session = onnxruntime.InferenceSession(temp_model.SerializeToString())
                ort_inputs = {inp: tensor_values[inp] for inp in node.input if inp in tensor_values}
                outputs = ort_session.run(None, ort_inputs)
                for i, output_name in enumerate(node.output):
                    tensor_values[output_name] = outputs[i]
            except Exception as e:
                # If onnxruntime fails, raise to fall back to llvmcpu
                print(f"Warning: Could not execute {node.op_type} with onnxruntime: {e}")
                raise

    # Collect output values
    outputs = []
    for output in graph.output:
        if output.name in tensor_values:
            outputs.append(tensor_values[output.name])

    return outputs


@versioned_unhashable_object_fixture
def numpy_reference_results(request, onnx_model_file, input_data):
    """Generate reference using numpy for bf16 MatMul (onnxruntime doesn't support it).
    Falls back to llvmcpu if numpy cannot handle an operation."""
    onnx_model = onnx.load(str(onnx_model_file))

    if not _has_bf16_matmul(onnx_model) or not _has_bf16_einsum(onnx_model):
        try:
            ort_session = onnxruntime.InferenceSession(str(onnx_model_file))
            ort_inputs = {inp.name: input_data[i] for i, inp in enumerate(ort_session.get_inputs())}
            return ort_session.run(None, ort_inputs)
        except Exception:
            pass
    try:
        return _execute_onnx_model_numpy(onnx_model, input_data)
    except Exception as e:
        # Fall back to llvmcpu if numpy cannot handle the operation
        llvmcpu_reference_results = request.getfixturevalue("llvmcpu_reference_results").data
        print(f"Warning: Numpy reference failed, falling back to llvmcpu: {e}")
        return llvmcpu_reference_results
