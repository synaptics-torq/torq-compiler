#!/bin/bash

# This scripts build libc, libm and compiler-rt for the CSS subsystem

set -e

function usage {
    echo "Usage: $0 <iree-source-dir> <build-dir>"
    echo "  <iree-build-dir>      : Path to the prebuilt build directory"
    echo "  <runtimes-build-dir>      : Path to the runtimes build directory"
}

if [[ ! -d "$1" ]] ; then
    usage 
    exit 1
fi

if [[ -z "$2" ]] ; then
    usage
    exit 1
fi

BASE_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")/..

IREE_SOURCE_DIR=${BASE_DIR}/third_party/iree

IREE_BUILD_DIR=$(readlink -f $1)

RUNTIMES_BUILD_DIR=$(readlink -f $2)

echo "Configuring host build to build hdrgen"

cd ${IREE_SOURCE_DIR}/third_party/llvm-project/llvm

cmake -B ${RUNTIMES_BUILD_DIR}/host \
      -G Ninja \
      -DLLVM_ENABLE_PROJECTS="libc" \
      -DCMAKE_C_COMPILER=${IREE_BUILD_DIR}/third_party/iree/llvm-project/bin/clang \
      -DCMAKE_CXX_COMPILER=${IREE_BUILD_DIR}/third_party/iree/llvm-project/bin/clang++ \
      -DLLVM_LIBC_FULL_BUILD=ON \
      -DLLVM_INCLUDE_DOCS=OFF \
      -DLLVM_INCLUDE_TESTS=OFF \
      -DLIBC_HDRGEN_ONLY=ON \
      -DCMAKE_BUILD_TYPE=Release

echo "Build libc-hdrgen"

cmake --build ${RUNTIMES_BUILD_DIR}/host --target libc-hdrgen

echo "Configuring target build to build libc and compiler-rt"

cd ${IREE_SOURCE_DIR}/third_party/llvm-project/runtimes

cmake -B ${RUNTIMES_BUILD_DIR}/target \
      -G Ninja \
      -DIREE_BUILD_DIR=${IREE_BUILD_DIR} \
      -DCMAKE_TOOLCHAIN_FILE=${BASE_DIR}/scripts/toolchain.riscv.cmake \
      -DLLVM_ENABLE_RUNTIMES="libc;compiler-rt" \
      -DLLVM_LIBC_FULL_BUILD=ON \
      -DTORQ_CSS=ON \
      -DLLVM_INCLUDE_DOCS=OFF \
      -DCOMPILER_RT_BAREMETAL_BUILD=ON \
      -DTEST_COMPILE_ONLY=ON \
      -DLLVM_DEFAULT_TARGET_TRIPLE=riscv32-none-elf \
      -DLIBC_TARGET_TRIPLE=riscv32-none-elf \
      -DLLVM_INCLUDE_TESTS=OFF \
      -DLIBC_HDRGEN_EXE=${RUNTIMES_BUILD_DIR}/host/bin/libc-hdrgen \
      -DCMAKE_BUILD_TYPE=Release

cmake --build ${RUNTIMES_BUILD_DIR}/target

cp ${RUNTIMES_BUILD_DIR}/target/compiler-rt/lib/generic/libclang_rt.builtins-riscv32.a ${BASE_DIR}/third_party/runtimes-prebuilt/
cp ${RUNTIMES_BUILD_DIR}/target/libc/lib/libc.a ${BASE_DIR}/third_party/runtimes-prebuilt/
cp ${RUNTIMES_BUILD_DIR}/target/libc/lib/libm.a ${BASE_DIR}/third_party/runtimes-prebuilt/
