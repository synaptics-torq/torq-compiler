#!/bin/bash

# This scripts creates a Python3 venv and install requirements for building and testing

# abort on error
set -e

function usage() {
    echo "usage: $0 PATH/TO/VENV PATH/TO/IREE_BUILD"
    echo  "  or, in case you want to specify a version of python:"
    echo "usage: $0 PATH/TO/VENV PATH/TO/IREE_BUILD PYTHON_CMD"
}

if [[ -z "$1" ]] ; then
    usage
    exit 1
fi

if [[ -z "$2" ]] ; then
    usage
    exit 1
fi

if [[ -z "$3" ]] ; then
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

# install requirements
# these following lines must be in sync with the contents of assemble_release_docker.sh
pip install -r requirements.txt
pip install third_party/iree/integrations/tensorflow/python_projects/iree_tf
pip install third_party/iree/integrations/tensorflow/python_projects/iree_tflite

# pre-register build outputs in virtual env
cat > $(python -c "import site; print(site.getsitepackages()[0])")/iree.pth << EOF
${BUILD_DIR}/third_party/iree/compiler/bindings/python
${BUILD_DIR}/third_party/iree/runtime/bindings/python
EOF

# register library code in virtual env
cat > $(python -c "import site; print(site.getsitepackages()[0])")/torq.pth << EOF
${BASE_DIR}/python
EOF
