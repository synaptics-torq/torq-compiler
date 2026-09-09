# Quickstart

```{important}
The following section assumes you are familiar with IREE. To get started with IREE you can follow the [IREE TensorFlowLite Guide](https://iree.dev/guides/ml-frameworks/tflite/)
and the [IREE CPU Deployment Guide](https://iree.dev/guides/deployment-configurations/cpu/).
```
## Setup

### Python Wheel (pip)

```{note}
The compiler and runtime are distributed as separate wheels in the GitHub release assets. Install both when compiling and running models on the host.
```

Download the matching compiler and runtime wheels from the [GitHub Releases](https://github.com/synaptics-torq/torq-compiler/releases) page. For example, replace `<version>` with the release version and run:

```bash
$ curl -LO https://github.com/synaptics-torq/torq-compiler/releases/download/<version>/torq_compiler-<version>-cp312-cp312-manylinux_2_28_x86_64.whl
$ curl -LO https://github.com/synaptics-torq/torq-compiler/releases/download/<version>/torq_runtime-<version>-cp312-cp312-manylinux_2_28_x86_64.whl
```

Install the downloaded wheels with `pip`:

```bash
$ pip install torq_compiler-<version>-cp312-cp312-manylinux_2_28_x86_64.whl
$ pip install torq_runtime-<version>-cp312-cp312-manylinux_2_28_x86_64.whl
```

Use the `manylinux_2_28_aarch64` `torq-runtime` wheel when installing the runtime on a supported aarch64 board. The compiler wheel is currently provided for x86-64 hosts.

The `torq-compiler` wheel provides `torq-compile` and the compiler Python tools. The `torq-runtime` wheel provides `torq-run-module`, the runtime Python API, and the host simulator where supported.

To enable ONNX model importing, install with the `onnx` extra:

```bash
$ pip install "torq_compiler-<version>-<platform>.whl[onnx]"
```

To enable TFLite model conversion, install with the `tflite` extra:

```bash
$ pip install "torq_compiler-<version>-<platform>.whl[tflite]"
```

To enable TensorFlow SavedModel importing, install the `tf` extra:

```bash
$ pip install "torq_compiler-<version>-<platform>.whl[tf]"
```

Multiple extras can be combined:

```bash
$ pip install "torq_compiler-<version>-<platform>.whl[onnx,tflite]"
```

For ONNX importing via Python:

```bash
$ python -m iree.compiler.tools.import_onnx model.onnx -o model.mlir
```

For TFLite conversion:

```bash
$ tosa-converter-for-tflite model.tflite --bytecode -o model.mlirbc
```

## Compile and Run the Model

- Download the MobileNetV2 INT8 MLIR model from the [Synaptics Hugging Face repository](https://huggingface.co/Synaptics/MobileNetV2):
    ```bash
    $ curl -L https://huggingface.co/Synaptics/MobileNetV2/resolve/main/MobileNetV2_int8.mlir?download=true -o MobileNetV2_int8.mlir
    ```

- Compile the downloaded MLIR model to a Torq runtime executable:
    ```bash
    $ torq-compile MobileNetV2_int8.mlir -o mobilenetv2_int8.vmfb
    ```

- Run the generated model with the Torq simulator:
    ```bash
    $ torq-run-module --module=mobilenetv2_int8.vmfb --input="1x224x224x3xi8=1"
    ```