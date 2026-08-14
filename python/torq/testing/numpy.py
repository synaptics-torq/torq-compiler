import numpy as np
import onnx
import onnxruntime

from .versioned_fixtures import versioned_unhashable_object_fixture

# The pure numpy/ONNX reference implementations live in torq.lab.reference.
# They are imported here so the pytest fixtures below can delegate to them and so
# torq.testing.onnx (and its test consumers) keep importing them from this module.
from torq.lab.reference import (  # noqa: F401
    torch_tanh_gelu_numpy,
    _has_bf16_matmul,
    _has_bf16_einsum,
    _has_gelu,
    _numpy_maxpool,
    _numpy_global_average_pool,
    _execute_onnx_model_numpy,
    _parse_instance_norm_params,
    _numpy_instance_norm,
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


@versioned_unhashable_object_fixture
def numpy_global_average_pool_reference_results(input_data):
    """Compute GlobalAveragePool reference in numpy with FP32 accumulation.

    For bf16 inputs, numpy promotes to float32 for accumulation
    (same as np.matmul behavior), matching the TORQ NPU.
    """
    data = input_data.data if hasattr(input_data, 'data') else input_data
    assert len(data) == 1
    return [_numpy_global_average_pool(data[0])]


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
    scale, bias, eps = _parse_instance_norm_params(mlir_model_file)
    return [_numpy_instance_norm(data[0], scale, bias, eps, output_dtype)]


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
