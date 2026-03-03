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
   $ echo $CR_PAT | docker login ghcr.io -u $GITHUB_USERNAME --password-stdin
   ```

   You can use your GitHub username and a GitHub personal *classic* access token as password.
   The access token must be configured with the following permissions: read:packages, repo
   The $CR_PAT varible must be set to the github access token.

   Please refer to [Github documentation for the creation and usage of a
   personal access token](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/managing-your-personal-access-tokens#creating-a-personal-access-token-classic).
   
   Please refer to the official docker documentation to install docker on your machine.
   Some hints for linux, Windows and MacOS are also available in the [SyNAP guide](https://synaptics-synap.github.io/doc/v/latest/docs/manual/working_with_models.html#installing-docker).

2. Start a development container with access to the current directory and your ssh configuration:

   **Note:** To build and mount volumes correctly, Docker needs access to the entire `torq-compiler` project directory. Running this command from the parent directory allows Docker to mount the full project directory inside the container.

   ```{code} shell
   $ cd .. && torq-compiler/scripts/dev.sh
   ```

   In alternative you can customize the docker execution with an alias such as the one in the example here below:

   ```{code} shell
   $ alias torq-dev='docker run -it --rm -u $(id -u):$(id -g) -v $MOUNT_PATH:$MOUNT_PATH -w $(pwd) -e PATH=$VENV_PATH:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin -e CCACHE_DIR=$CCACHE_PATH -e HOME=$HOME_PATH -e IREE_BUILD_DIR=$BUILD_PATH -e ADB_SERVER_SOCKET=tcp:host.docker.internal:5037 ghcr.io/synaptics-torq/torq-compiler-dev/builder'
   ```

   where the variables have the following meaning:
   ``$MOUNT_PATH`` root directory to be mounted inside the docker container
   ``$VENV_PATH`` path to the ``bin`` directory inside the python virtual env to be used
   ``CCACHE_PATH`` path to the ``ccache`` working directory
   ``HOME_PATH`` path to the home directory (some utilities store info inside the home dir)
   ``BUILD_PATH`` path to the build directory (if not using the default ``iree-build``)

   The alias can then be used directly to start the torq development environment:

   ```{code} shell
   $ torq-dev
   ```
   
   Some tests download items from HuggingFace, in order for this to work you have to login
   to HuggingFace as well. This has to be done only once using an HuggingFace token,
   the registration information will be stored inside the ``$HOME_PATH`` directory:
   
   ```{code} shell
   hf auth login --token $HF_PAT
   ```
   
   where ``$HF_PAT`` is the personal access token received from HuggingFace.
   
   The docker comes with ``adb`` preinstalled and it is possible to use it to connect
   to an astra board via TCP/IP or USB.

3. Inside the container, go to the `torq-compiler` directory and continue with the build steps:

   ```{code} shell
   $ cd torq-compiler
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

    If running inside a docker be sure the CCACHE_DIR is configured correctly, eg:
    ```{code} shell
    export CCACHE_DIR=../iree-build/ccache
    ```
    This can be configured when the docker image is started using the `-e` option.

    Initialize the ccache:
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
    $ scripts/configure_soc_build.sh ../iree-build-soc ../iree-build astra_machina poky
    ```

4. Run the cross-compile build:

    ```{code} shell
    $ cmake --build ../iree-build-soc/ --target torq-run-module
    ```

The statically linked ``torq-run-module`` is available in ``../iree-build-soc/third_party/iree/tools/torq-run-module``.
