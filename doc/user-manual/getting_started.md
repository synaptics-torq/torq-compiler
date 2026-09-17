# Quickstart

```{important}
The following section assumes you are familiar with IREE. To get started with IREE you can follow the [IREE TensorFlowLite Guide](https://iree.dev/guides/ml-frameworks/tflite/)
and the [IREE CPU Deployment Guide](https://iree.dev/guides/deployment-configurations/cpu/).
```
## Setup

The `torq-compiler` and `torq-runtime` are provided as Python wheels for Linux (x86-64, and aarch64 for the runtime).

- **Linux x86-64 (e.g. Ubuntu, or an aarch64 board)**: skip directly to [Python Wheel (pip)](#python-wheel-pip) below.
- **macOS**: Native macOS support is planned for a future release. No compatible wheel exists, so first follow [macOS](#macos) to set up a Linux container, then continue with [Python Wheel (pip)](#python-wheel-pip) inside it.

### macOS

[Colima](https://github.com/abiosoft/colima) provides a lightweight Docker environment on macOS.

- Install Docker and Colima:
    ```bash
    $ brew install docker colima
    ```

    ```{note}
    Requires [Homebrew](https://brew.sh) to be installed. If you don't have it, follow the installation instructions on the official Homebrew page.
    ```

- Start the lightweight virtual machine that powers your local Docker environment:
    ```bash
    $ colima start --cpu 4 --memory 4
    ```

    ```{note}
    Adjust `--cpu` and `--memory` to match your available resources and model size; `colima start` alone defaults to 2 CPUs and 2 GiB of memory, which may be too little for compiling larger models.
    ```

- Launch an Ubuntu container, mounting your current directory so files are shared between the host and the container:
    ```bash
    $ docker run --rm -it --platform linux/amd64 -v $(pwd):$(pwd) -w $(pwd) -u root:$(id -g) ubuntu:24.04 bash
    ```

    ```{note}
    `--platform linux/amd64` is required because the `torq` wheels are currently only published for x86-64. `-u root:$(id -g)` runs the container as `root` but with your host user's group ID, so files created in the mounted directory keep a group your host user can read/write instead of being owned by an arbitrary container group.
    ```

- Inside the container, install Python and other required packages:
    ```bash
    $ apt-get update && apt-get install -y curl
    $ apt install -y python3 python3-pip python3-venv
    $ python3 --version
    Python 3.12.3
    ```

- Create and activate a Python virtual environment:
    ```bash
    $ python3 -m venv myenv && source myenv/bin/activate
    ```

    ```{note}
    A virtual environment isolates the packages installed for this project from the container's system-wide Python packages. Recent Ubuntu/Debian releases mark the system Python as "externally managed" and refuse `pip install` outside of a virtual environment, so creating one with `venv` avoids conflicts and keeps installs reproducible.
    ```

With the virtual environment active, continue with the [Python Wheel (pip)](#python-wheel-pip) steps below to install and use `torq-compiler` and `torq-runtime` inside the container.

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