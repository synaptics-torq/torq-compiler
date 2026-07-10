#!/bin/bash

# Adds static release files and downloads HF models into a pre-assembled release directory.
# The release/ directory must already contain merged host and astra packages.
# Usage: assemble_release_docker.sh <install-dir>

set -e

BASE_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")/..

if [[ -z "$1" ]]; then
    echo "Usage: $0 <install-dir>"
    exit 1
fi

INSTALL_DIR=$(readlink -f $1)

mkdir -p ${INSTALL_DIR}/scripts

# Static release files from source tree
cp ${BASE_DIR}/scripts/Dockerfile.release  ${INSTALL_DIR}/Dockerfile
cp ${BASE_DIR}/scripts/setup.sh            ${INSTALL_DIR}/
cp ${BASE_DIR}/scripts/apt-packages.txt    ${INSTALL_DIR}/

cp ${BASE_DIR}/scripts/diff-tensor.py          ${INSTALL_DIR}/scripts/
cp ${BASE_DIR}/scripts/image_to_tensor.py      ${INSTALL_DIR}/scripts/
cp ${BASE_DIR}/scripts/annotate_profiling.py   ${INSTALL_DIR}/scripts/

cp ${BASE_DIR}/pytest.ini   ${INSTALL_DIR}/
cp -r ${BASE_DIR}/tests     ${INSTALL_DIR}/

# Download release models from HuggingFace
python3 ${BASE_DIR}/scripts/model_release.py ${INSTALL_DIR}/tests

# Remove __pycache__
find ${INSTALL_DIR} -type d -name "__pycache__" -prune -exec rm -rf {} \;
