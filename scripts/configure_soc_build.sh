#!/bin/bash

# This scripts creates a build directory for IREE with the Synaptics Torq driver suitable for running on targets

function usage () {
    echo "$0 <soc-fpga-build-dir> <host-build-dir> [target] [toolchain]"
}

if [[ -z "$1" ]] ; then
    usage
    exit 1
fi

if [[ -z "$2" ]] ; then
    usage
    exit 1
fi

BASE_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")/..
SOC_BUILD_DIR=$(readlink -f $1)
HOST_BUILD_DIR=$(readlink -f $2)
TARGET=${3:-soc_fpga}

# Set CMake flags based on target
if [[ "$TARGET" == "astra_machina" ]]; then
    TORQ_ENABLE_SOC_FPGA=OFF
    TORQ_ENABLE_ASTRA_MACHINA=ON

    if [[ "$4" == "poky" ]]; then
        # Use the Poky toolchain
        SDK_DIR="/opt/synaptics/astra/toolchain"
        echo "Sourcing Astra SDK environment..."
        . "${SDK_DIR}/environment-setup-cortexa55-poky-linux"
        TOOLCHAIN_FILE="${SDK_DIR}/sysroots/x86_64-pokysdk-linux/usr/share/cmake/cortexa55-poky-linux-toolchain.cmake"
        PYTHON3="${SDK_DIR}/sysroots/x86_64-pokysdk-linux/usr/bin/python3"
        PYVER="$("${PYTHON3}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    else
        # Default to aarch64 toolchain
        TOOLCHAIN_FILE=${BASE_DIR}/scripts/toolchain.aarch64.cmake
    fi
else
    TORQ_ENABLE_SOC_FPGA=ON
    TORQ_ENABLE_ASTRA_MACHINA=OFF
    TOOLCHAIN_FILE=${BASE_DIR}/scripts/toolchain.armhf.cmake
fi

cd ${BASE_DIR}

CMAKE_ARGS=(
  -B "${SOC_BUILD_DIR}"
  -G Ninja
  -DIREE_HOST_BIN_DIR="${HOST_BUILD_DIR}/third_party/iree/tools"
  -DCMAKE_BUILD_TYPE=Release
  -DIREE_BUILD_COMPILER=OFF
  -DTORQ_ENABLE_AWS_FPGA=OFF
  -DTORQ_ENABLE_SOC_FPGA="${TORQ_ENABLE_SOC_FPGA}"
  -DTORQ_ENABLE_ASTRA_MACHINA="${TORQ_ENABLE_ASTRA_MACHINA}"
  -DTORQ_ENABLE_SIMULATOR=OFF
  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON
  -DIREE_HAL_DRIVER_LOCAL_TASK=ON
  -DIREE_ENABLE_CPUINFO=OFF
  -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}"
)

if [[ "$TARGET" == "astra_machina" && "$4" == "poky" ]]; then
  CMAKE_ARGS+=(
    -DIREE_BUILD_PYTHON_BINDINGS=ON
    -DPython3_EXECUTABLE="${PYTHON3}"
    -DPython3_INCLUDE_DIR="${SDK_DIR}/sysroots/cortexa55-poky-linux/usr/include/python${PYVER}"
    -DPython_INCLUDE_DIR="${SDK_DIR}/sysroots/cortexa55-poky-linux/usr/include/python${PYVER}"
    -DPython_NumPy_INCLUDE_DIR="${SDK_DIR}/sysroots/cortexa55-poky-linux/usr/lib/python${PYVER}/site-packages/numpy/core/include"
    -DPython3_NumPy_INCLUDE_DIR="${SDK_DIR}/sysroots/cortexa55-poky-linux/usr/lib/python${PYVER}/site-packages/numpy/core/include"
  )
else
  CMAKE_ARGS+=(-DIREE_BUILD_PYTHON_BINDINGS=OFF)
fi

echo "Configuring build for target: $TARGET"
cmake "${CMAKE_ARGS[@]}"
