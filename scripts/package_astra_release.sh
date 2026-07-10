#!/bin/bash

# Packages Astra target build outputs into a release/ directory structure.
# Usage: package_astra_release.sh <target-build-dir> <install-dir>

set -e

TARGET_BUILD_DIR=$(readlink -f $1)
INSTALL_DIR=$2

mkdir -p ${INSTALL_DIR}/tools ${INSTALL_DIR}/lib

cp ${TARGET_BUILD_DIR}/runtime/tools/torq-run-module \
   ${INSTALL_DIR}/tools/astra-sl-torq-run-module

cp ${TARGET_BUILD_DIR}/third_party/iree/runtime/plugins/TORQ/torq_hw/hal/SL2610/syna_npu.ko \
   ${INSTALL_DIR}/lib/syna_npu.ko

# Strip astra binary using cross-strip from the toolchain
CROSS_STRIP=$(find /opt/synaptics/astra/toolchain -name "*-strip" -executable 2>/dev/null | head -1)
${CROSS_STRIP:-strip} ${INSTALL_DIR}/tools/astra-sl-torq-run-module 2>/dev/null || true
