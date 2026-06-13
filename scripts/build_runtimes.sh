#!/bin/bash

# This script builds the prebuilt runtime libraries that live in
# `third_party/runtimes-prebuilt/`. Both archives are tracked via Git LFS;
# this script is for maintainers to (re)build them when the IREE submodule
# (and therefore its bundled LLVM) is updated.
#
# Supported architectures:
#   - riscv32  →  libclang_rt.builtins-riscv32.a, libc.a, libm.a   (CSS subsystem)
#   - x86_64   →  libclang_rt.builtins-x86_64.a                    (host CPU dispatches)

set -e

function usage {
    echo "Usage: $0 <arch> <iree-build-dir> <runtimes-build-dir>"
    echo "  <arch>                : 'riscv32' or 'x86_64'"
    echo "  <iree-build-dir>      : Path to the IREE build dir (needs IREE-bundled clang)"
    echo "  <runtimes-build-dir>  : Path to the runtimes build scratch dir"
}

if [[ -z "$1" ]] || [[ ! -d "$2" ]] || [[ -z "$3" ]] ; then
    usage
    exit 1
fi

ARCH="$1"
BASE_DIR=$(dirname "$(realpath "${BASH_SOURCE[0]}")")/..
IREE_SOURCE_DIR=${BASE_DIR}/third_party/iree
IREE_BUILD_DIR=$(readlink -f "$2")
RUNTIMES_BUILD_DIR=$(readlink -f "$3")

case "$ARCH" in
    riscv32)
        TOOLCHAIN_FILE="${BASE_DIR}/scripts/toolchain.riscv.cmake"
        ENABLE_RUNTIMES="libc;compiler-rt"
        TARGET_TRIPLE="riscv32-none-elf"
        SUB_BUILD="target-riscv"
        EXTRA_FLAGS=(
            -DLLVM_LIBC_FULL_BUILD=ON
            -DLIBC_CONF_THREAD_MODE=LIBC_THREAD_MODE_SINGLE
            -DLIBC_TARGET_TRIPLE="$TARGET_TRIPLE"
        )
        ARCHIVES=(
            "compiler-rt/lib/generic/libclang_rt.builtins-riscv32.a"
            "libc/lib/libc.a"
            "libc/lib/libm.a"
        )
        ;;
    x86_64)
        TOOLCHAIN_FILE="${BASE_DIR}/scripts/toolchain.x86.cmake"
        ENABLE_RUNTIMES="compiler-rt"
        TARGET_TRIPLE="x86_64-unknown-elf"
        SUB_BUILD="target-x86"
        EXTRA_FLAGS=(
            -DCOMPILER_RT_BUILD_BUILTINS=ON
            -DCOMPILER_RT_BUILD_SANITIZERS=OFF
            -DCOMPILER_RT_BUILD_XRAY=OFF
            -DCOMPILER_RT_BUILD_LIBFUZZER=OFF
            -DCOMPILER_RT_BUILD_PROFILE=OFF
            -DCOMPILER_RT_BUILD_MEMPROF=OFF
            -DCOMPILER_RT_BUILD_ORC=OFF
        )
        ARCHIVES=(
            "compiler-rt/lib/generic/libclang_rt.builtins-x86_64.a"
        )
        ;;
    *)
        echo "Unknown arch: '$ARCH' (expected 'riscv32' or 'x86_64')"
        usage
        exit 1
        ;;
esac

echo "Configuring target build for $ARCH (runtimes: $ENABLE_RUNTIMES)"

cd "${IREE_SOURCE_DIR}/third_party/llvm-project/runtimes"

cmake -B "${RUNTIMES_BUILD_DIR}/${SUB_BUILD}" \
      -G Ninja \
      -DIREE_BUILD_DIR="${IREE_BUILD_DIR}" \
      -DCMAKE_TOOLCHAIN_FILE="${TOOLCHAIN_FILE}" \
      -DLLVM_ENABLE_RUNTIMES="${ENABLE_RUNTIMES}" \
      -DLLVM_INCLUDE_DOCS=OFF \
      -DCOMPILER_RT_BAREMETAL_BUILD=ON \
      -DTEST_COMPILE_ONLY=ON \
      -DLLVM_DEFAULT_TARGET_TRIPLE="${TARGET_TRIPLE}" \
      -DLLVM_INCLUDE_TESTS=OFF \
      -DCMAKE_BUILD_TYPE=Release \
      "${EXTRA_FLAGS[@]}"

cmake --build "${RUNTIMES_BUILD_DIR}/${SUB_BUILD}"

for archive in "${ARCHIVES[@]}"; do
    cp "${RUNTIMES_BUILD_DIR}/${SUB_BUILD}/${archive}" \
       "${BASE_DIR}/third_party/runtimes-prebuilt/"
done

echo "Archives staged in ${BASE_DIR}/third_party/runtimes-prebuilt/:"
for archive in "${ARCHIVES[@]}"; do
    echo "  - $(basename ${archive})"
done
