# Step-by-Step Model Deployment Examples

This guide walks through converting, compiling, and running ML models on a Torq device.

## Prerequisites

- Activate the Python environment as explained in [Getting Started](./getting_started.md). (Skip this step if using the Docker container.)

- Navigate to the root directory of the [Release Package](./getting_started.md#release-package-ubuntu-2404), or run the [Docker container](./getting_started.md#docker-image).
  For the Docker container, the release package is located at:
  ```{code}shell
  $ cd /opt/release
  ```

## Prepare Model (Convert to MLIR)

Before compiling, your model must be converted to MLIR. See [Model Preparation](./model_conversion.md) for full instructions on each framework.

Here is a quick summary of conversion commands:

**TFLite:**
```{code}shell
$ iree-import-tflite path/to/model.tflite -o model.tosa
```

**ONNX:**
```{code}shell
$ python -m iree.compiler.tools.import_onnx path/to/model.onnx -o model.mlir --data-prop
```

**Torch:**
See [Convert Torch Model to MLIR](./model_conversion.md#convert-torch-model-to-mlir) for the Python-based conversion workflow.

```{note}
The `tests/hf/` directory containing sample models is only included in the [Release Package](./getting_started.md#release-package-ubuntu-2404) and is not available in the compiler GitHub repository.
```

## Compile for Device

Use `torq-compile` to compile the MLIR model into a Torq runtime executable (`.vmfb`).

The key flags are:

| Flag | Description |
|------|-------------|
| `--iree-input-type=<type>` | Specifies the MLIR dialect of the input file. See table below. |
| `-o <file>.vmfb` | Output path for the compiled model. |

**Input types by source framework:**

| Source Framework | Input Type | Typical File Extension |
|-----------------|------------|----------------------|
| Auto (default) | `auto` | all |
| TFLite | `tosa-torq` | `.tosa` |
| ONNX | `onnx-torq` | `.mlir` |
| Torch | `torch-torq` | `.mlir` |
| Linalg | `linalg-torq` | `.mlir` |

```{note}
Input type defaults to `auto`, which defers input type detection and conversion to the compiler. This behavior can be overridden by manually specifying `--iree-input-type`.
```

### Example: TFLite model (MobileNetV2)

**Model Source:** The MobileNetV2 model is generated from tf.keras.applications using [tf_model_generator.py](https://github.com/synaptics-torq/iree-synaptics-synpu/blob/main/tests/model_generator/tf_model_generator.py). The dataset used for int8 quantization consists of random data. This model is only included in the [Release Package](./getting_started.md#release-package-ubuntu-2404) and is not available in the compiler GitHub repository.

```{code}shell
# Convert TFLite to TOSA
$ iree-import-tflite tests/hf/Synaptics_MobileNetV2/MobileNetV2_int8.tflite -o mobilenetv2.tosa

# Compile for device
$ torq-compile mobilenetv2.tosa -o mobilenetv2.vmfb
```

### Example: ONNX model

```{code}shell
# Convert ONNX to MLIR
$ python -m iree.compiler.tools.import_onnx path/to/model.onnx -o model.mlir --data-prop

# Compile for device
$ torq-compile model.mlir -o model.vmfb
```

## Run Inference on Torq

Use the compiled model with `torq-run-module` to run inference:

```{code}shell
$ torq-run-module \
    --module=mobilenetv2.vmfb \
    --function=main \
    --input="1x224x224x3xi8=1"
```

```{note}
To use an actual image as input, preprocess the image and save it as a NumPy array (`.npy` file). Then provide the path to the `.npy` file as input, as described in the [Input Guide](./input_guide.md#converting-images-to-npy-or-bin).
```

### Using the Python Bindings

You can also run inference programmatically using the `torq-runtime` Python package:

```python
from torq.runtime import VMFBInferenceRunner

runner = VMFBInferenceRunner("mobilenetv2.vmfb", device_uri="torq")
outputs = runner.infer(inputs)
print(f"Inference took {runner.infer_time_ms} ms")
```

The `VMFBInferenceRunner` class supports options such as `function` and `load_method`. See the [Torq Runtime Python API](./torq_runtime.md) page for the full API reference.

## Compiling for Host (Simulation)

To compile for the current host machine (e.g., for testing without device hardware), add simulator-specific flags:

```{code}shell
$ torq-compile mobilenetv2.tosa \
    --torq-css-qemu \
    --torq-target-host-triple=native \
    -o mobilenetv2.vmfb
```

## Handling Unsupported Operations

For information about handling unsupported operations and CSS fallback, see [Handling Unsupported Ops](./css_host_fallback.md).