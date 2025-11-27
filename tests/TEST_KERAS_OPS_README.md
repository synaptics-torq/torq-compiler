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
pytest test_keras_ops.py::test_keras_model[model050_mult_inp1x10x10x4_int8-cmodel] -v
```

### List Tests Without Running
```bash
pytest test_keras_ops.py --collect-only
pytest test_keras_ops.py -m "depthwise and int8" --collect-only
```

## Available Markers

**Operation Types:**
`add_mul_sub`, `conv`, `conv_transpose`, `depthwise`, `fc`, `mean`, `pointwise`, `pooling`, `activation`, `softmax`, `transpose`

**Quantization:**
`int8`, `int16`

View all markers:
```bash
pytest --markers
```
