#!/bin/bash

set -e

BASE_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")/..

MPACT_DIR=${BASE_DIR}/third_party/coralnpu-mpact/x86_64-Linux

mkdir -p ${MPACT_DIR}

echo "Downloading MPACT artifacts to ${MPACT_DIR}..."

${BASE_DIR}/scripts/download_artifact.py --output-dir ${MPACT_DIR} --glob coralnpu-mpact-simulator*.tar.gz

echo "Extracting MPACT artifacts to ${MPACT_DIR}..."

tar -xzf ${MPACT_DIR}/coralnpu-mpact-simulator*.tar.gz --strip-components=1 -C ${MPACT_DIR}

rm ${MPACT_DIR}/coralnpu-mpact-simulator*.tar.gz
