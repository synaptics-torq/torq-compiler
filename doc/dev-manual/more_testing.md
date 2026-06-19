# More testing

This page documents opt-in test suites. They are not run by default or in CI, and only run
when you opt in explicitly (e.g. with a marker). Use them for coverage or investigation work
that is too large or slow to run every time. Add new opt-in suites here as their own section.

## Convolution sweep

`tests/test_conv_sweep.py` is a parametric convolution sweep. Rather than hand-writing conv
test cases or pulling them out of existing models, it samples a space of geometries (kernel,
stride, padding, channels, spatial size, for both Conv1D and Conv2D) and builds a model for
each one in memory at collection time. It covers two backends:

- TFLite, built from `tf.keras` (`conv_model` in `tests/models/keras_models.py`), in three
  flavors: `f32`, `w8if32` (dynamic-range), and `w8i8` (full int8).
- ONNX, built from `torch` (`conv_onnx_model` in `tests/models/conv_sweep.py`), in `f32` and
  `bf16`.

`conv_sweep_params()` in `tests/models/conv_sweep.py` produces the geometries and maps each
to a `ConvModelParams`. Each one becomes a `Case` that goes through the usual `case_config`
-> model -> MLIR -> compile -> run pipeline. Nothing is written to the repository; the
intermediate `.tflite`, `.onnx`, `.mlir`, and `.vmfb` files go to the pytest cache.

The sweep carries the `conv_sweep` marker and is excluded from the default run and CI
(`pytest.ini` sets `-m "not conv_sweep"` in `addopts`). Opt in by selecting the marker — an
explicit `-m` on the command line overrides the default:

```
# run the whole sweep (numeric where possible, compile-only otherwise)
pytest tests/test_conv_sweep.py -m conv_sweep

# filter to a backend / flavor / geometry via -k
pytest tests/test_conv_sweep.py -m conv_sweep -k "conv_tflite_w8i8"

# compile-coverage only (the compile-only test function)
pytest tests/test_conv_sweep.py -m conv_sweep -k test_conv_sweep_compile -rA
```

Each case runs either `test_conv_sweep_numeric` (compile, run, compare against the backend
oracle) or `test_conv_sweep_compile` (compile only). Two flavors are checked numerically
today: TFLite `w8i8`, whose int8 I/O matches the oracle, and ONNX `bf16`, which has native
`bf16` I/O and gets its oracle from the numpy/torch reference path. The rest are compile
only: TFLite `f32`/`w8if32` and ONNX `f32` need `--torq-convert-io-dtype`, which rewrites the
compiled model's I/O to `bf16` while the harness still feeds and compares `f32`. Numeric
verification for those depends on harness work that is in progress elsewhere.

To extend coverage, add or widen a category in `CATEGORIES` (or the sampling ranges) in
`tests/models/conv_sweep.py`. No new test code or committed model files are needed.
