# Synaptics Torq Compiler

The `torq-compiler` Python package enables the compilation of machine learning models for the Synaptics Torq NPU and provides simulation capabilities for corresponding hardware. It bundles the Torq compiler alongside the IREE compiler Python bindings into a single wheel with both namespaces.

## Installation

```bash
pip install torq-compiler
```

## Optional extras

Install with one or more extras to enable optional features (e.g. `pip install "torq-compiler[onnx,tflite]"`):

- `onnx`: ONNX model import, self-verification, `torq-gen-config`, and quantization
- `tflite`: TFLite model conversion via `tosa-converter-for-tflite`
- `tf`: TensorFlow SavedModel import and TFLite dynamic-to-static shape conversion (`torq-convert-static`)
- `profile`: profiling annotation and Perfetto trace rendering
- `all`: all of the extras above

## Documentation

Comprehensive compiler documentation is available in the [Usage Guide](https://synaptics-torq.github.io/torq-compiler/v/latest/).
