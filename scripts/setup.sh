#!/bin/bash

# This script installs all system dependencies for Ubuntu 24.04 and sets up the torq-compiler environment

# abort on error
set -e

function usage() {
    echo "usage: $0 PATH/TO/VENV"
}

if [[ -z "$1" ]] ; then
    echo "Please provide the path to the venv directory: $0 ../venv"
    exit 1
fi

# Configurable venv path
BASE_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")
VENV_DIR=$(readlink -f $1)


# create a new virtual env if one doesn't exist yet
if [[ ! -d ${VENV_DIR} ]]; then
    python3 -m venv ${VENV_DIR}
fi

# Set PYTHONPATH
if [[ ! -d "$BASE_DIR/python/runtime" || ! -d "$BASE_DIR/python/compiler" ]]; then
  echo "Missing $BASE_DIR/python/runtime or $BASE_DIR/python/compiler"
  exit 1
fi

PY_RUNTIME="$(realpath "$BASE_DIR/python/runtime")"
PY_COMPILER="$(realpath "$BASE_DIR/python/compiler")"

ACTIVATE_FILE="$VENV_DIR/bin/activate"

# activate the virtual env
source $VENV_DIR/bin/activate

# pre-register build outputs in virtual env
cat > $($VENV_DIR/bin/python -c "import site; print(site.getsitepackages()[0])")/iree.pth << EOF
${PY_RUNTIME}
${PY_COMPILER}
EOF

# Install Python dependencies
$VENV_DIR/bin/pip install -r $BASE_DIR/python/requirements.txt
$VENV_DIR/bin/pip install -e $BASE_DIR/python/iree_tf
$VENV_DIR/bin/pip install -e $BASE_DIR/python/iree_tflite

# [FIX-ME - # this is an hack until we don't fix the RPATH in the binaries]
# Allow the compiler and simulator to find required libraries
if ! grep -q "export LD_LIBRARY_PATH=${BASE_DIR}/lib" "$ACTIVATE_FILE"; then
  echo "export LD_LIBRARY_PATH=${BASE_DIR}/lib:\$LD_LIBRARY_PATH" >> "$ACTIVATE_FILE"
fi

# Add tools and scripts directories to PATH
if ! grep -q "export PATH=$BASE_DIR/tools:$BASE_DIR/scripts" "$ACTIVATE_FILE"; then
  echo "export PATH=$BASE_DIR/tools:$BASE_DIR/scripts:\$PATH" >> "$ACTIVATE_FILE"
fi
