#!/bin/bash

set -e

BASE_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")/..

cd $BASE_DIR

EXTRA_HEADER=$(git config --get http.https://github.com/.extraheader || true)

# append github token to command lines if present in the main repo
if [[ -n "${EXTRA_HEADER}" ]] ; then
    USE_EXTRA_HEADER=(-c "http.extraheader=${EXTRA_HEADER}")
else
    USE_EXTRA_HEADER=()
fi
echo "Updating third_party/iree submodule..."

git "${USE_EXTRA_HEADER[@]}" submodule update --depth=1 --init third_party/iree

cd third_party/iree

echo "Updating submodules of third_party/iree..."

git "${USE_EXTRA_HEADER[@]}" submodule sync

git "${USE_EXTRA_HEADER[@]}" submodule update --depth=1 --init third_party/benchmark \
                                                        third_party/cpuinfo \
                                                        third_party/flatcc \
                                                        third_party/googletest \
                                                        third_party/llvm-project \
                                                        third_party/tracy \
                                                        third_party/pybind11 \
                                                        third_party/torch-mlir
