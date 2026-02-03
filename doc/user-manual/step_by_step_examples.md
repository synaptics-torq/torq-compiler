# Step-by-Step Model Deployment Examples

## Converting and Running MobileNetV2 on Torq

Follow the steps below to convert, and run a MobileNetV2 model on Torq.

### Set up environment

- If not yet done, activate Python environment as explained in [Getting Started](./getting_started.md). (Skip this step if using the Docker container)

### Convert TFLite → TOSA

- Navigate to the root directory of the [Release Package (Ubuntu 24.04)](./getting_started.md#release-package-ubuntu-24-04), or run the [Docker container](./getting_started.md#docker-image).
  For the Docker container, release package is located at:  
  ```
  $ cd /opt/release
  ```
- Convert the `.tflite` model into TOSA format using the IREE import tool:

  **Model Source:** This model - MobileNetV2 is generated from tf.keras.applications using [tf_model_generator.py](https://github.com/synaptics-torq/iree-synaptics-synpu/blob/main/tests/model_generator/tf_model_generator.py). The dataset for int8 quantization is done using random data.

  ```shell
  # Convert TFLite to TOSA (binary MLIR)
  iree-import-tflite tests/hf/Synaptics_MobileNetV2/MobileNetV2_int8.tflite -o mobilenetv2.tosa
  ```
> **Note:** The `tests/hf/` directory is only included in the [Release Package](./getting_started.md#release-package-ubuntu-24-04) and is not available in the compiler github repository.

  The TOSA format is a binary format that can be fed to our compiler. For more details on the IREE TFLite tools, see the [official IREE website](https://iree.dev/guides/ml-frameworks/tflite/).

- Optionally convert the TOSA file (binary MLIR) to the text representation:

  ```shell
  # Convert binary MLIR to text MLIR
  $ iree-opt mobilenetv2.tosa -o mobilenetv2.mlir
  ```

  While torq-compile supports TOSA binary format it is useful at this early stage of the compiler development to use the MLIR text representation in order to facilitate debugging and error reporting.

### Compile TOSA → Torq model

- Convert the TOSA file into a Torq runtime executable (`.vmfb`):
  ```shell
  $ torq-compile mobilenetv2.tosa -o mobilenetv2.vmfb
  ```
  or (when using the text MLIR format)
  ```shell
  $ torq-compile mobilenetv2.mlir -o mobilenetv2.vmfb
  ```

### Run inference on Torq

- Use the compiled model with the Torq runtime to run inference. For example:
  ```shell
  $ iree-run-module \
    --device=torq \
    --module=mobilenetv2.vmfb \
    --function=main \
    --input="1x224x224x3xi8=1"
  ```

> **Note:** To use an actual image as input, preprocess the image and save it as a NumPy array (`.npy` file). Then, provide the path to the `.npy` file as input, as described in the [Input Guide](./input_guide.md#converting-images-to-npy-or-bin).