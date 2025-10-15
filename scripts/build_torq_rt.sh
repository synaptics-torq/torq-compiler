#!/bin/bash

set -x

BASE_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
PDIR=${BASE_DIR}/..
SRC_DIR=${PDIR}/torq-hw
BUILD_DIR=$1
INSTALL_DIR=$2
cmake -S ${SRC_DIR} -B ${BUILD_DIR} -G Ninja
cmake --build ${BUILD_DIR}
cmake --install ${BUILD_DIR} --prefix ${INSTALL_DIR}
