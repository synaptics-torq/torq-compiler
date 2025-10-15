#!/bin/bash

# This scripts creates a build directory for IREE with the Synaptics Torq driver

if [[ -z "$1" ]] ; then
    echo "Please provide the path to the build directory: $0 <build-dir>"
    exit 1
fi

BASE_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")/..

BUILD_DIR=$(readlink -f $1)

cd $BASE_DIR

# check if SCCACHE is installed
if command -v sccache &> /dev/null
then
    echo "Found sccache, enabling it in the build"
    LAUNCHER_OPTS="-DCMAKE_C_COMPILER_LAUNCHER=sccache -DCMAKE_CXX_COMPILER_LAUNCHER=sccache"
# check if ccache is installed
elif command -v ccache &> /dev/null
then
    echo "Found ccache, enabling it in the build"
    LAUNCHER_OPTS="-DCMAKE_C_COMPILER_LAUNCHER=ccache -DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
else
    LAUNCHER_OPTS=""
fi

# sudo apt install python3-numpy python3-pybind11 install nanobind-dev
# 
#  -G Ninja
# 
#     Enables a build using Ninja that is faster than the default based on make
#
# -DCMAKE_BUILD_TYPE=RelWithDebInfo 
#
#     Set up a release build with debug information. This is done to be able to analyze stack
#     traces while ensuring the performance of the program is not too slow.
#
#  -DCMAKE_C_COMPILER=clang and -DCMAKE_CXX_COMPILER=clang++
# 
#     Enables build with our preferred toolchain
#
#  -DIREE_BUILD_PYTHON_BINDINGS=ON
#
#     Build the python bindings
#
#  -DPython3_EXECUTABLE="$(which python3)"
#
#     Use the currently active python interpreted (uses the interpreter of a virtualenv if that is enabled)
#
# -DIREE_ENABLE_SPLIT_DWARF=ON and -DIREE_ENABLE_THIN_ARCHIVES=ON
#     
#     Speed up build by reducing unnecessary I/O
# 
#  -DIREE_ENABLE_LLD=ON
#
#     Use lld, required when building with clang (see later)
#
#  -DCMAKE_C_COMPILER=clang and -DCMAKE_CXX_COMPILER=clang++
# 
#     Enables build with our preferred toolchain
#
#  -DIREE_HAL_DRIVER_LOCAL_SYNC=ON and -DIREE_HAL_DRIVER_LOCAL_TASK=ON
#
#     Enable CPU drivers in the runtime in addition to the Torq HAL for tests
#
#  -DIREE_BUILD_PYTHON_BINDINGS=ON
#
#     Build the python bindings
#
#  -DPython3_EXECUTABLE="$(which python3)"
#
#     Use the currently active python interpreted (uses the interpreter of a virtualenv if that is enabled)
#
#  -DTORQ_ENABLE_SIMULATOR=ON
#
#     Enable building cmodel test vector runtime (for testing purposes)
#

python3 ./tests/helpers/py_version_checker.py

cmake -B $BUILD_DIR \
            -G Ninja \
            -DCMAKE_BUILD_TYPE=RelWithDebInfo \
            -DCMAKE_C_COMPILER=clang \
            -DCMAKE_CXX_COMPILER=clang++ \
            -DIREE_ENABLE_SPLIT_DWARF=ON \
            -DIREE_ENABLE_THIN_ARCHIVES=ON \
            -DIREE_ENABLE_LLD=ON \
            -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
            -DIREE_HAL_DRIVER_LOCAL_TASK=ON \
            ${LAUNCHER_OPTS} \
            -DIREE_BUILD_PYTHON_BINDINGS=ON \
            -DTORQ_ENABLE_SIMULATOR=ON \
            -DPython3_EXECUTABLE="$(which python3)"
