#!/bin/bash

# Packages host build outputs into a release/ directory structure.
# Stripping is done here so the tarball is already minimal.
# Usage: package_host_release.sh <build-dir> <install-dir>

set -e

BASE_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")/..

BUILD_DIR=$(readlink -f $1)
INSTALL_DIR=$2

mkdir -p ${INSTALL_DIR}/tools ${INSTALL_DIR}/lib \
         ${INSTALL_DIR}/python/compiler ${INSTALL_DIR}/python/runtime \
         ${INSTALL_DIR}/scripts

# Host tools
cp ${BUILD_DIR}/third_party/iree/tools/iree-run-module     ${INSTALL_DIR}/tools/
cp ${BUILD_DIR}/third_party/iree/tools/iree-compile        ${INSTALL_DIR}/tools/
cp ${BUILD_DIR}/third_party/iree/tools/iree-c-embed-data   ${INSTALL_DIR}/tools/
cp ${BUILD_DIR}/third_party/iree/tools/iree-flatcc-cli     ${INSTALL_DIR}/tools/
cp ${BUILD_DIR}/third_party/iree/tools/torq-compile        ${INSTALL_DIR}/tools/
cp ${BUILD_DIR}/third_party/iree/tools/iree-opt            ${INSTALL_DIR}/tools/
cp ${BUILD_DIR}/third_party/torq-hw/rt/torq_rt_cm          ${INSTALL_DIR}/tools/
cp ${BUILD_DIR}/runtime/tools/torq-run-module              ${INSTALL_DIR}/tools/

# Host libs — libIREECompiler.so + mpact .so (copied into lib/ by workflow before calling this script)
cp ${BUILD_DIR}/third_party/iree/lib/libIREECompiler.so    ${INSTALL_DIR}/lib/
cp ${BUILD_DIR}/third_party/iree/lib/libcoralnpu_simulator_mpact.so ${INSTALL_DIR}/lib/

# Python bindings from build outputs
cp -rL ${BUILD_DIR}/third_party/iree/compiler/bindings/python/iree ${INSTALL_DIR}/python/compiler/
cp -rL ${BUILD_DIR}/third_party/iree/runtime/bindings/python/iree  ${INSTALL_DIR}/python/runtime/

# Python packages from source tree (iree submodule checked out by build job)
cp -rL ${BASE_DIR}/python/torq ${INSTALL_DIR}/python/
cp -r ${BASE_DIR}/third_party/iree/integrations/tensorflow/python_projects/iree_tf     ${INSTALL_DIR}/python/
cp -r ${BASE_DIR}/third_party/iree/integrations/tensorflow/python_projects/iree_tflite ${INSTALL_DIR}/python/
cp ${BASE_DIR}/requirements.txt ${INSTALL_DIR}/python/requirements.txt

# Remove redundant copy of libIREECompiler.so from python bindings
rm -f ${INSTALL_DIR}/python/compiler/iree/compiler/_mlir_libs/libIREECompiler.so

# Strip host binaries and libraries
find ${INSTALL_DIR}/lib   -name "*.so"        -exec strip {} \; 2>/dev/null || true
find ${INSTALL_DIR}/tools -type f -executable -exec strip {} \; 2>/dev/null || true

# Remove __pycache__
find ${INSTALL_DIR} -type d -name "__pycache__" -prune -exec rm -rf {} \;
