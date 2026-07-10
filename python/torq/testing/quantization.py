import onnx
from onnx import numpy_helper, TensorProto

from .versioned_fixtures import versioned_hashable_object_fixture
from .dtype_utils import cast_round_trip, ConvertIODTypesPolicy


def pytest_addoption(parser):
    parser.addoption(
        "--fake-quantize",
        action="store_true",
        default=False,
        help="Fake quantize FP32/INT64 weights to BF16/INT32",
    )


# ---- ONNX quantization helpers ---- #

@versioned_hashable_object_fixture
def onnx_fake_quantize_config(request):
    """Version key so caches invalidate when ``--fake-quantize`` changes.

    Fake-quantization rounds FP32/INT64 weights to BF16/INT32.  It must be part
    of the model-file cache key so the compiled Torq path and the ONNX
    reference path are always derived from the *same* quantized weights, even
    when the cache is reused (CI runs without ``--recompute-cache``).
    """
    return {"fake_quantize": bool(request.config.getoption("--fake-quantize", default=False))}


def onnx_fake_quantize(model: onnx.ModelProto) -> onnx.ModelProto:
    for init in model.graph.initializer:
        if init.data_type not in (TensorProto.FLOAT, TensorProto.INT64, TensorProto.UINT64):
            continue
        data = numpy_helper.to_array(init).copy()
        down_dtype = ConvertIODTypesPolicy.convert_io_dtype(data.dtype)
        new_data = cast_round_trip(data, down_dtype, data.dtype)
        new_init = numpy_helper.from_array(new_data, init.name)
        init.CopyFrom(new_init)
    try:
        model = onnx.shape_inference.infer_shapes(model)
    except Exception:
        pass
    return model
