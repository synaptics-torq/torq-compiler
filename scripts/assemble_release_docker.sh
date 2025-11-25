#!/bin/bash

# This script assembles in a directory the files necessary
# to create a docker image with runtime and compiler
#

set -e

function usage() {
    echo "Usage: $0 <build-dir> <target-build-dir> <install-dir>"
    echo "  build-dir: Build dir containing the host version of the compiler/runtime"
    echo "  target-build-dir: Build dir containing the target version of the runtime."
    echo "  install-dir: Directory where to assemble files."
}

BASE_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")/..

if [[ -z "$1" ]]; then
    usage
    exit 1
fi

if [[ -z "$2" ]]; then
    usage
    exit 1
fi

if [[ -z "$3" ]]; then
    usage
    exit 1
fi

BUILD_DIR=$1
TARGET_BUILD_DIR=$2
INSTALL_DIR=$3

if [[ -d ${INSTALL_DIR} ]]; then
    echo "Directory ${INSTALL_DIR} already exists. Please remove it first."
    exit 1
fi

mkdir ${INSTALL_DIR}

mkdir ${INSTALL_DIR}/tools
mkdir ${INSTALL_DIR}/lib
mkdir ${INSTALL_DIR}/scripts

cp ${BUILD_DIR}/third_party/iree/tools/iree-run-module ${INSTALL_DIR}/tools
cp ${BUILD_DIR}/third_party/iree/tools/iree-compile ${INSTALL_DIR}/tools
cp ${BUILD_DIR}/third_party/iree/tools/iree-opt ${INSTALL_DIR}/tools
cp ${BUILD_DIR}/third_party/iree/lib/libIREECompiler.so ${INSTALL_DIR}/lib

cp ${TARGET_BUILD_DIR}/third_party/iree/tools/iree-run-module ${INSTALL_DIR}/tools/armhf-iree-run-module

cp ${BASE_DIR}/scripts/Dockerfile.release ${INSTALL_DIR}/Dockerfile     
cp ${BASE_DIR}/scripts/setup.sh ${INSTALL_DIR}
cp ${BASE_DIR}/scripts/apt-packages.txt ${INSTALL_DIR}

cp ${BASE_DIR}/scripts/torq-compile ${INSTALL_DIR}/scripts
cp ${BASE_DIR}/scripts/diff-tensor.py ${INSTALL_DIR}/scripts
cp ${BASE_DIR}/scripts/image_to_tensor.py ${INSTALL_DIR}/scripts
cp ${BASE_DIR}/scripts/annotate_profiling.py ${INSTALL_DIR}/scripts

mkdir ${INSTALL_DIR}/python

mkdir -p ${INSTALL_DIR}/python ${INSTALL_DIR}/python/compiler ${INSTALL_DIR}/python/runtime
cp -rL ${BUILD_DIR}/third_party/iree/compiler/bindings/python/iree ${INSTALL_DIR}/python/compiler
cp -rL ${BUILD_DIR}/third_party/iree/runtime/bindings/python/iree ${INSTALL_DIR}/python/runtime
cp -rL ${BASE_DIR}/python/torq ${INSTALL_DIR}/python

cat ${BASE_DIR}/third_party/iree/runtime/bindings/python/iree/runtime/build_requirements.txt \
    ${BASE_DIR}/third_party/iree/integrations/tensorflow/test/requirements.txt \
    ${BASE_DIR}/python/torq/requirements.txt \
    ${BASE_DIR}/requirements.txt > ${INSTALL_DIR}/python/requirements.txt

cp -r ${BASE_DIR}/third_party/iree/integrations/tensorflow/python_projects/iree_tf ${INSTALL_DIR}/python
cp -r ${BASE_DIR}/third_party/iree/integrations/tensorflow/python_projects/iree_tflite ${INSTALL_DIR}/python

mkdir ${INSTALL_DIR}/samples

# Download release models from HF
python3 ${BASE_DIR}/scripts/model_release.py ${INSTALL_DIR}/samples

# remove redundant libIREECompiler.so library
rm ${INSTALL_DIR}/python/compiler/iree/compiler/_mlir_libs/libIREECompiler.so

# remove all __pycache__ directories
find ${INSTALL_DIR} -type d -name "__pycache__" -prune -exec rm -rf {} \;

# strip all the binaries and libraries
find ${INSTALL_DIR} -name "*.so" -exec strip {} \;
find ${INSTALL_DIR}/tools -type f -executable ! -name "armhf-*" -exec strip {} \;
