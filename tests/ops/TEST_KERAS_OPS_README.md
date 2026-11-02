# Keras Operations Test Suite

## Usage

### Run All Tests
```bash
pytest test_keras_ops.py -v
```

### Filter by Operation
```bash
pytest test_keras_ops.py -m depthwise -v        # Depthwise convolution
pytest test_keras_ops.py -m conv -v             # Convolution
pytest test_keras_ops.py -m fc -v               # Fully connected
pytest test_keras_ops.py -m pooling -v          # Pooling
pytest test_keras_ops.py -m activation -v       # Activation functions
```

### Filter by Quantization
```bash
pytest test_keras_ops.py -m int8 -v             # Int8 quantization only
pytest test_keras_ops.py -m int16 -v            # Int16 quantization only
```

### Combine Filters
```bash
pytest test_keras_ops.py -m "depthwise and int8" -v           # Depthwise with int8
pytest test_keras_ops.py -m "(conv or depthwise) and int16" -v # Conv or depthwise with int16
pytest test_keras_ops.py -m "not pooling" -v                   # Everything except pooling
```

### Run Specific Test
```bash
pytest "test_keras_ops.py::test_keras_model_torq[model050_mult_inp1x10x10x4_int8-sim-sl2610-v1]" -v
```

### List Tests Without Running
```bash
pytest test_keras_ops.py --collect-only
pytest test_keras_ops.py -m "depthwise and int8" --collect-only
```

## Available Markers

**Operation Types:**
one per family module in `tests/testdata/keras_ops/` (the module name): `add_mul_sub`, `conv`, `conv_transpose`, `depthwise`, `fc`, `mean`, `pointwise`, `pooling`, `resize`, `activation`, `softmax`, `transpose`, `conv2d_layers`, `nnr301_convtranspose`, `audio_vision_depthwise`

**Quantization:**
`int8`, `int16`

View all markers:
```bash
pytest --markers
```

## Adding Test Cases

The models are defined in `tests/testdata/keras_ops/<family>.py`; the module name is the marker of
its cases. Each module has a `CASES` dict from a case name to a model builder, written with
`torq.testing.keras_builder`:

```python
from torq.testing.keras_builder import conv2d, layers, sequential

CASES = {
    # one layer, built after tf.keras.utils.set_random_seed(42)
    "maxpool_k3_s2_16x16x8": sequential((16, 16, 8), layers.MaxPooling2D(pool_size=3, strides=2)),
    # a single Conv2D (with bias): shape, filters, kernel size, strides, padding
    "conv_k3_s2_32x32x16": conv2d((32, 32, 16), 8, (3, 3), (2, 2), "same"),
    # a sweep
    **{f"conv_k{k}_s{s}_32x32x16": conv2d((32, 32, 16), 8, (k, k), (s, s))
       for k in (1, 3, 5) for s in (1, 2)},
}
```

Each case runs with int8 and int16 quantization, as `<case>_int8` and `<case>_int16`.
`functional(shape, build)` builds functional-API models (see `add_mul_sub.py`). The optional
`CONFIG` dict sets a case's quantizations, compiler options and comparison tolerances (see
`tests/ops/keras_ops_models.py`). Cases that must not run at all are listed in
`keras_known_failures.py`; expected failures go in `test_keras_ops-xfails.txt`.

### Cases ported from tosa_ops

The `tosa_*` modules hold cases ported from `tests/testdata/tosa_ops`: each keeps the name of the
MLIR file it replaced and its quantization (`QUANTIZE = "int8"`, with a few int16 cases in
`CONFIG`), and `DEFAULT_CONFIG` makes them run on the slices only, as `test_tosa_ops` does. Port a
tosa_ops case only when Keras and TFLite produce the same TOSA ops and attributes (pads, strides,
perms, axes); keep the MLIR file when it tests explicit pads, constants, scales or zero points, an
op TFLite never emits, or bf16.

