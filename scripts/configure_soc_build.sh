#!/bin/bash

# This scripts creates a build directory for IREE with the Synaptics Torq driver suitable for running on targets

function usage () {
    echo "$0 <soc-fpga-build-dir> <host-build-dir> [target]"
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
    TOOLCHAIN_FILE=toolchain.aarch64.cmake
else
    TORQ_ENABLE_SOC_FPGA=ON
    TORQ_ENABLE_ASTRA_MACHINA=OFF
    TOOLCHAIN_FILE=toolchain.armhf.cmake
fi

cd ${BASE_DIR}

echo "Configuring build for target: $TARGET"

cmake -B ${SOC_BUILD_DIR} \
            -G Ninja \
            -DIREE_HOST_BIN_DIR=${HOST_BUILD_DIR}/third_party/iree/tools \
            -DCMAKE_BUILD_TYPE=Release \
            -DIREE_BUILD_COMPILER=OFF \
            -DTORQ_ENABLE_AWS_FPGA=OFF \
            -DTORQ_ENABLE_SOC_FPGA=${TORQ_ENABLE_SOC_FPGA} \
            -DTORQ_ENABLE_ASTRA_MACHINA=${TORQ_ENABLE_ASTRA_MACHINA} \
            -DTORQ_ENABLE_SIMULATOR=OFF \
            -DCMAKE_TOOLCHAIN_FILE=${BASE_DIR}/scripts/${TOOLCHAIN_FILE}
