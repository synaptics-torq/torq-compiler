# Quickstart
------------

```{important}
The following section assumes you are familiar with IREE. To get started with IREE you can follow the [IREE TensorFlowLite Guide](https://iree.dev/guides/ml-frameworks/tflite/)
and the [IREE CPU Deployment Guide](https://iree.dev/guides/deployment-configurations/cpu/).
```
## Setup

- Install compiler and simulator in **one of the following** two ways.

### Release Package Ubuntu 24.04

- Uncompress the release package (tarball or zip).
- There are some prerequisite system packages that need to be installed.
Please refer to apt-packages.txt for the list.
- In the root directory of the uncompressed package, run:
    ```bash
    $ ./setup.sh <venv-directory>
    ```
    This will:
    - Create a Python virtual environment at `<venv-directory>`.
    - Install all required Python dependencies, including IREE compiler and runtime.
    - Set up import tools and other necessary components.
    > **Note:** The setup process may take some time as it installs all dependencies and tools.

- Once setup is complete, activate the Python environment:
    ```bash
    $ source <venv-directory>/bin/activate
    ```
    You can now use the compiler and runtime tools from this environment.

### Docker Image

You can use **either** of the following approaches:

**A. Use the prebuilt image:**
- Create an ephemeral Docker container that uses the prebuilt image:
    ```bash
    $ docker run --rm -it -v $(pwd):$(pwd) -w $(pwd) -u $(id -u):$(id -g) ghcr.io/syna-astra-dev/iree-synaptics-synpu/compiler:latest
    ```
    The container will have access to the contents of your current directory.

**B. Build your own image (if you don't have access to the Synaptics GitHub repo):**
- Uncompress the [Release Package Ubuntu 24.04](#release-package-ubuntu-24-04).
- In the root directory of the uncompressed package, build the Docker image using the provided Dockerfile:
    ```bash
    $ docker build -t <image-name> .
    ```
- Run the Docker container:
    ```bash
    $ docker run --rm -it -v $(pwd):$(pwd) -w $(pwd) -u $(id -u):$(id -g) <image-name>
    ```
## Compile and Run the Model

- Example MLIR models are provided in the `samples/` directory in the package.

    - **[Release Package](#release-package-ubuntu-24-04):**  
    Navigate to the `samples/` directory is located in the root of the uncompressed package.

    - **[Docker Image](#docker-image):**  
    The `samples/` directory is located in the `/opt/release` directory.
    You can navigate there with:
        ```
        $ cd /opt/release
        ```

- Compile an input MLIR file ``samples/tosa/conv2d-stride4.mlir`` to a compiled model ``model.vmfb``:
    ```bash
    $ torq-compile samples/tosa/conv2d-stride4.mlir -o model.vmfb
    ```

- Run the generated model with the Torq simulator:
    ```bash
    $ iree-run-module --device=torq --module=model.vmfb --input="1x256x256x1xi8=0"
    ```