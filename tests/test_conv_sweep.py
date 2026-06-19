"""Parametric convolution sweep (opt-in via the `conv_sweep` marker).

Single in-framework entry point for the generic-conv geometry sweep. Models are
built in memory at collection time (keras->TFLite and torch->ONNX) and run
through the existing Case/fixture pipeline; nothing is written to the repo.

The sweep is heavy, so it carries the `conv_sweep` marker and is excluded from
default and CI collection by `-m "not conv_sweep"` in pytest.ini's addopts. Run it
explicitly with `pytest -m conv_sweep` (an explicit `-m` overrides the default).
Each case runs either numerically (vs the backend oracle) or
compile-only. Numeric today: TFLite w8i8 (int8 I/O matches the oracle) and ONNX
bf16 (native bf16 I/O, oracle via the numpy/torch path). Compile-only: the
f32-activation flavors (TFLite f32/w8if32, ONNX f32) -- they need
--torq-convert-io-dtype, which rewrites the vmfb I/O to bf16 while the harness
oracle/input stay f32 (the io-dtype reconciliation is being handled separately).

Reuses, end to end: the model builders in models/conv_sweep.py (conv_model,
conv_onnx_model, conv_sweep_tflite_model_file), tflite_reference_results, the
onnx_model->mlir chain + composite_reference_results oracle, torq_compiled_model /
torq_results, Case.
"""

import pytest

from torq.testing.cases import Case
from torq.testing.comparison import compare_test_results

from .models.conv_sweep import (                                     # noqa: F401 (fixtures)
    conv_sweep_params, conv_model, keras_model_params,
    conv_onnx_model, conv_onnx_dtype,
    conv_sweep_tflite_model_file, conv_sweep_flavor,
)

# NSS-only so a conv that silently falls back to host/CSS fails loudly.
NSS_ONLY = ["--torq-disable-host", "--torq-disable-css"]
# f32-activation flavors must lower to the bf16 conv ALU.
DTYPE_CONVERT = ["--torq-convert-dtypes", "--torq-convert-io-dtype"]

# (backend, flavor, compile_only). A flavor must be compile_only whenever it
# needs --torq-convert-io-dtype (which rewrites the vmfb I/O to bf16) -- the
# harness feeds an f32 oracle/input, so numeric comparison hits
# hal.buffer_view.assert (tensor element type mismatch). Numerically runnable:
# native-int8 w8i8 (int8 I/O matches the TFLite oracle) and ONNX bf16 (already a
# bf16 model -> native bf16 I/O, no io-dtype convert; oracle via the bf16 numpy/
# torch path in composite_reference_results). The f32/w8if32 flavors still need
# the convert flags and stay compile-only until the io-dtype harness work lands.
_VARIANTS = [
    ("tflite", "w8i8",   False),  # numeric: int8 I/O matches oracle
    ("tflite", "w8if32", True),   # io-dtype convert -> compile-only
    ("tflite", "f32",    True),   # io-dtype convert -> compile-only
    ("onnx",   "f32",    True),   # io-dtype convert -> compile-only
    ("onnx",   "bf16",   False),  # numeric: native bf16 I/O, no convert needed
]


def get_conv_sweep_cases():
    cases = []
    for p in conv_sweep_params():
        for backend, flavor, compile_only in _VARIANTS:
            opts = list(NSS_ONLY)
            # f32-activation flavors need the bf16 lowering flags; bf16 (already a
            # bf16 model) and native-int8 w8i8 compile as-is. Matches the original
            # compile-coverage study's flavor handling.
            if flavor in ("f32", "w8if32"):
                opts += DTYPE_CONVERT
            data = {
                "backend": backend,
                "flavor": flavor,
                "compile_only": compile_only,
                "keras_model_params": p,
                "torq_compiler_options": opts,
            }
            if backend == "tflite":
                data.update({
                    "keras_model": "conv_model",
                    "mlir_model_file": "tflite_mlir_model_file",
                    # sweep-local flavor converter (f32 / w8if32 / w8i8); the
                    # shared framework has no dynamic-range path.
                    "tflite_model_file": "conv_sweep_tflite_model_file",
                    "input_data": "tweaked_random_input_data",
                })
            else:  # onnx
                data.update({
                    "onnx_model": "conv_onnx_model",
                    "onnx_dtype": flavor,
                    "mlir_model_file": "onnx_mlir_model_file",
                    "input_data": "tweaked_random_input_data",
                })
            name = f"conv_{backend}_{flavor}_{p.idfn()}"
            cases.append(Case(name, data))
    return cases


def pytest_generate_tests(metafunc):
    if "case_config" not in metafunc.fixturenames:
        return
    # Cases are always generated (Case objects are cheap; models are built lazily by
    # fixtures at run time). The `conv_sweep` marker on the tests keeps them out of
    # default/CI collection -- see the module docstring.
    cases = get_conv_sweep_cases()
    metafunc.parametrize("case_config", cases, indirect=True, ids=[c.name for c in cases])


@pytest.fixture
def case_config(request):
    return request.param.data


@pytest.mark.conv_sweep
def test_conv_sweep_numeric(request, case_config, torq_results):
    if case_config["compile_only"]:
        pytest.skip("compile-only flavor (bf16 io-dtype harness gap)")
    if case_config["backend"] == "tflite":
        reference = request.getfixturevalue("tflite_reference_results")
    else:
        reference = request.getfixturevalue("composite_reference_results")
    compare_test_results(request, torq_results, reference, case_config)


@pytest.mark.conv_sweep
def test_conv_sweep_compile(case_config, torq_compiled_model):
    if not case_config["compile_only"]:
        pytest.skip("numeric flavor runs in test_conv_sweep_numeric")
