#!/bin/bash

#!/bin/bash

# Minimal Python setup for Astra builder image
# Creates a venv, installs Astra-specific Python requirements.

# abort on error
set -e

function usage() {
    echo "usage: $0 PATH/TO/VENV PATH/TO/IREE_BUILD"
    echo "  Optionally specify Python executable as third arg: $0 VENV BUILD PYTHON_CMD"
}

if [[ -z "$1" ]] || [[ -z "$2" ]]; then
    usage
    exit 1
fi

if [[ -z "$3" ]]; then
    PYTHON_CMD=python3
else
    PYTHON_CMD=$3
    echo "using ${PYTHON_CMD}"
fi

BASE_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")/..

# create a new virtual env if one doesn't exist yet
if [[ ! -d $1 ]]; then
    ${PYTHON_CMD} -m venv $1
fi

VENV_DIR=$(readlink -f $1)
BUILD_DIR=$(readlink -f $2)

# activate the virtual env
source $VENV_DIR/bin/activate

cd $BASE_DIR

pip install -r requirements_astra.txt || {
    echo "Dependency resolution failed. Check requirements.txt for version conflicts."
    exit 1
}
