#!/bin/bash

set -x

BASE_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
PLUGIN_DIR=${BASE_DIR}/..
SRC_DIR=${PLUGIN_DIR}/../iree
BUILD_DIR=$1
INSTALL_DIR=$2
cmake -S ${SRC_DIR} -B ${BUILD_DIR} -G Ninja \
            -DIREE_CMAKE_PLUGIN_PATHS=${PLUGIN_DIR} \
            -DCMAKE_BUILD_TYPE=Debug \
            -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
            -DIREE_BUILD_COMPILER=OFF \
            -DIREE_ERROR_ON_MISSING_SUBMODULES=OFF \
            -DIREE_ENABLE_ASSERTIONS=ON \
            -DIREE_ENABLE_SPLIT_DWARF=ON \
            -DIREE_ENABLE_THIN_ARCHIVES=ON \
            -DIREE_EXTERNAL_HAL_DRIVERS=torq \
            -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
            -DIREE_TARGET_BACKEND_LLVM_CPU=ON \
            -DIREE_HAL_DRIVER_DEFAULTS=OFF \
            -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
            -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
            -DIREE_INPUT_STABLEHLO=OFF \
            -DCMAKE_C_COMPILER=clang \
            -DCMAKE_CXX_COMPILER=clang++

cmake --build ${BUILD_DIR} --target iree-run-module 

# unfortunately cmake install will trigger build of all targets (various test and test
#cmake --install ${BUILD_DIR} --prefix ${INSTALL_DIR}

mkdir -p ${INSTALL_DIR}/bin
cp ${BUILD_DIR}/third_party/iree/tools/iree-run-module ${INSTALL_DIR}/bin/
