# Build and setup

## Prerequisites

The recommended development is a native Ubuntu 24.04 machine or WSL 2 with an Ubuntu 24.04 image.

## Download the code

1. Ensure you have both `git` and `git-lfs` pre-installed on your machine:

   ```{code} shell
   $ git lfs --version
   ```

2. Clone the root repository:

   ```{code} shell
   $ git clone https://github.com/synaptics-torq/torq-compiler.git
   ```

3. Go to the directory that was cloned:

    ```{code} shell
    $ cd torq-compiler
    ```

4. Clone the required submodules (some submodules are not necessary for the build):

   ```{code} shell
   $ scripts/checkout_submodules.sh   
   ```

## Install required system packages

If you are using an Ubunutu 24.04 environment you can install system packages required for the build with the following command: 

   ```{code} shell
   $ scripts/install_dependencies.sh
   ```
If you are using a different environment you can use a Docker image:

1. Log-in to the GitHub docker registry

   ```{code} shell
   $ docker login ghcr.io
   ```

   Use your Github username and a Github personal access token as password.
   Please refer to [Github documentation for the creation and usage of a
   personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic).
   

2. Start a development container with access to the current directory and your ssh configuration:

   ```{code} shell
   $ scripts/dev.sh
   ```

## Build compiler and runtime for host

1. Setup a python virtual environment with the packages required for development:

   ```{code} shell
   $ scripts/configure_python.sh ../venv ../iree-build
   ```
   The first parameter is the location of the venv and the second is the location
   where the build will be performed so that the build outputs can be pre-enabled
   in the environment. 

2. Activate the environment:

   ```{code} shell
   $ source ../venv/bin/activate   
   ```

3. (optional but strongly suggested) Setup `ccache` as follows::

    ```{code} shell
    $ ccache --max-size=20G
    ```

4. Setup the build system with the following command line:

    ```{code} shell
    $ scripts/configure_build.sh ../iree-build
    ```

4. Build the Torq compiler and the runtime:

    ```{code} shell
    $ cmake --build ../iree-build/ --target torq
    ```

   Building IREE from source may take several hours, especially on a typical laptop, due to the project's size and complexity.

## Build runtime for target

In order to cross-compile the runtime for an embedded target use the following commands:

1. Build the host version of the compiler as explained in the previous section (some host tools 
   are required for the cross-build)

2. Configure the cross-compile build:

    ```{code} shell
    $ scripts/configure_soc_build.sh ../iree-build-soc ../iree-build
    ```

4. Run the cross-compile build:

    ```{code} shell
    $ cmake --build ../iree-build-soc/ --target iree-run-module
    ```

The statically linked ``iree-run-module`` is available in ``../iree-build-soc/third_party/iree/tools/iree-run-module``.