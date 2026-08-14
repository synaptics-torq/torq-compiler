from .versioned_fixtures import versioned_hashable_object_fixture


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
