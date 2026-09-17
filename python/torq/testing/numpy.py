import re
from pathlib import Path

import ml_dtypes
import numpy as np
import onnx
import onnxruntime
from onnx import numpy_helper, TensorProto

from .versioned_fixtures import versioned_unhashable_object_fixture

# The pure numpy/ONNX reference implementations live in torq.lab.verification.reference.
# They are imported here so the pytest fixtures below can delegate to them and so
# torq.testing.onnx (and its test consumers) keep importing them from this module.
from torq.lab.verification.reference import (  # noqa: F401
    torch_tanh_gelu_numpy,
    has_bf16_matmul,
    has_bf16_einsum,
    has_gelu,
    numpy_maxpool,
    numpy_global_average_pool,
    execute_onnx_model_numpy,
    parse_instance_norm_params,
    numpy_instance_norm,
)


@versioned_unhashable_object_fixture
def numpy_gelu_reference_results(input_data, mlir_io_spec):
    assert len(input_data) == 1
    assert len(mlir_io_spec.outputs) == 1

    from .iree import get_dtype

    output_dtype = get_dtype(mlir_io_spec.outputs[0].fmt)
    return [torch_tanh_gelu_numpy(input_data[0], output_dtype)]


@versioned_unhashable_object_fixture
def numpy_matmul_reference_results(input_data, mlir_io_spec):
    """Compute matmul reference in float32 then cast back to output dtype."""
    assert len(input_data) == 2
    assert len(mlir_io_spec.outputs) == 1

    from .iree import get_dtype

    output_dtype = get_dtype(mlir_io_spec.outputs[0].fmt)
    result = np.matmul(input_data[0], input_data[1])
    return [result.astype(output_dtype)]


def _parse_uniform_bf16_constant(mlir_model_file):
    """Extract (value, shape) from a uniform bf16 dense constant in an MLIR file.

    The const-weight matmul repro cases bake the weight as
    ``dense<6.250000e-02> : tensor<768x3072xbf16>``, so the golden can be
    computed directly in numpy instead of routing through the llvm-cpu
    reference (whose bf16 accumulation differs from the fp32 accumulation the
    NSS kernels actually use).
    """
    text = Path(str(mlir_model_file)).read_text()
    match = re.search(r"dense<([0-9.eE+-]+)>\s*:\s*tensor<([0-9x]+)xbf16>", text)
    if not match:
        raise ValueError(f"no uniform bf16 constant found in {mlir_model_file}")
    value = float(match.group(1))
    shape = [int(dim) for dim in match.group(2).split("x")]
    return value, shape


@versioned_unhashable_object_fixture
def numpy_matmul_const_weight_reference_results(input_data, mlir_io_spec, mlir_model_file):
    """Compute a uniform-const-weight matmul reference in fp32, round once to bf16.

    The model has a single external input (the activation); the weight is a
    uniform bf16 constant parsed out of the MLIR. Accumulating in fp32 and
    rounding the output once matches what the NSS matmul kernels do, so this is
    a genuine numpy golden for the K-split repros (whose bf16 chunk-boundary
    rounding is what we want to compare, not an llvm-cpu bf16-accumulated one).
    """
    from .iree import get_dtype

    data = input_data.data if hasattr(input_data, "data") else input_data
    assert len(data) == 1
    assert len(mlir_io_spec.outputs) == 1

    output_dtype = get_dtype(mlir_io_spec.outputs[0].fmt)
    value, shape = _parse_uniform_bf16_constant(mlir_model_file)
    weight = np.full(shape, value, dtype=np.float32)
    result = np.matmul(data[0].astype(np.float32, copy=False), weight)
    return [result.astype(output_dtype)]


@versioned_unhashable_object_fixture
def numpy_global_average_pool_reference_results(input_data):
    """Compute GlobalAveragePool reference in numpy with FP32 accumulation.

    For bf16 inputs, numpy promotes to float32 for accumulation
    (same as np.matmul behavior), matching the TORQ NPU.
    """
    data = input_data.data if hasattr(input_data, 'data') else input_data
    assert len(data) == 1
    return [numpy_global_average_pool(data[0])]


@versioned_unhashable_object_fixture
def numpy_instancenorm_reference_results(input_data, mlir_io_spec, mlir_model_file):
    """InstanceNormalization reference computed in numpy with fp32 accumulation.

    Bypasses the llvm-cpu reference, whose bf16-accumulated mean/variance is wildly
    inaccurate for large spatial reductions (and order-dependent), while the TORQ NPU
    accumulates in fp32.
    """
    from .iree import get_dtype

    data = input_data.data if hasattr(input_data, 'data') else input_data
    assert len(data) == 1
    assert len(mlir_io_spec.outputs) == 1
    output_dtype = get_dtype(mlir_io_spec.outputs[0].fmt)
    scale, bias, eps = parse_instance_norm_params(mlir_model_file)
    return [numpy_instance_norm(data[0], scale, bias, eps, output_dtype)]


@versioned_unhashable_object_fixture
def numpy_reference_results(request, onnx_model_file, input_data):
    """Generate reference using numpy for bf16 MatMul (onnxruntime doesn't support it).
    Falls back to llvmcpu if numpy cannot handle an operation."""
    onnx_model = onnx.load(str(onnx_model_file))

    if not has_bf16_matmul(onnx_model) or not has_bf16_einsum(onnx_model):
        try:
            ort_session = onnxruntime.InferenceSession(str(onnx_model_file))
            ort_inputs = {inp.name: input_data[i] for i, inp in enumerate(ort_session.get_inputs())}
            return ort_session.run(None, ort_inputs)
        except Exception:
            pass
    try:
        return execute_onnx_model_numpy(onnx_model, input_data)
    except Exception as e:
        # Fall back to llvmcpu if numpy cannot handle the operation
        llvmcpu_reference_results = request.getfixturevalue("llvmcpu_reference_results").data
        print(f"Warning: Numpy reference failed, falling back to llvmcpu: {e}")
        return llvmcpu_reference_results


def _cast_bf16_model_to_fp32(model):
    """Return a copy of an ONNX model with all bf16 initializers and tensor types
    cast to fp32 so onnxruntime (which has no bf16 kernels) can execute it. Used to
    build a reference for full bf16 models that ORT/numpy/llvm-cpu cannot run."""
    import copy

    m = copy.deepcopy(model)
    graph = m.graph
    for init in graph.initializer:
        if init.data_type == TensorProto.BFLOAT16:
            dims = list(init.dims)
            arr = np.frombuffer(init.raw_data, dtype=ml_dtypes.bfloat16).astype(np.float32)
            if dims:
                arr = arr.reshape(dims)
            init.CopyFrom(numpy_helper.from_array(arr, init.name))
    for vi in list(graph.input) + list(graph.output) + list(graph.value_info):
        if vi.type.HasField("tensor_type") and vi.type.tensor_type.elem_type == TensorProto.BFLOAT16:
            vi.type.tensor_type.elem_type = TensorProto.FLOAT
    for node in graph.node:
        if node.op_type == "Cast":
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.BFLOAT16:
                    attr.i = TensorProto.FLOAT
    return m


def _run_ort_fp32(onnx_model, input_data):
    """Cast a bf16 model to fp32 and run it through onnxruntime, casting inputs to
    fp32 as needed. Returns the outputs rounded back to bf16 precision so the
    comparison against the (bf16) torq result is bf16-vs-bf16."""
    fp32_model = _cast_bf16_model_to_fp32(onnx_model)
    ort_session = onnxruntime.InferenceSession(fp32_model.SerializeToString())
    ort_inputs = {}
    for i, inp in enumerate(ort_session.get_inputs()):
        data = input_data[i]
        if hasattr(data, "astype"):
            data = data.astype(np.float32)
        ort_inputs[inp.name] = data
    outputs = ort_session.run(None, ort_inputs)
    return [np.asarray(o).astype(ml_dtypes.bfloat16).astype(np.float32) for o in outputs]


@versioned_unhashable_object_fixture
def bf16_onnx_fp32_reference_results(mlir_model_file, input_data):
    """Reference for a bf16 torch_ops model whose ops the llvm-cpu/torch references
    cannot run (e.g. a full backbone whose bf16 MaxPool trips 'util.initializer'
    legalization on llvm-cpu). Requires a sibling <name>.onnx next to the .mlir: cast
    it to fp32, run onnxruntime, round back to bf16. Wired in tests/test_torch_ops.py."""
    onnx_path = Path(str(mlir_model_file)).with_suffix(".onnx")
    model = onnx.load(str(onnx_path))
    return _run_ort_fp32(model, input_data)
