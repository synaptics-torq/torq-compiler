#!/bin/bash
# Build a torq-compiler Python wheel.
#
# Usage:
#   $0 <host-build-dir>
#
# Example:
#   scripts/build_compiler_wheel.sh ../iree-build
#
# The finished wheel is placed in dist/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${PROJECT_DIR}/dist"

function usage () {
    echo "$0 <host-build-dir>"
}

if [[ -z "${1:-}" ]]; then
    echo "Error: host-build-dir is required." >&2
    usage
    exit 1
fi

HOST_BUILD_DIR="$(cd "${PROJECT_DIR}" && readlink -f "$1")"

if [[ ! -d "${HOST_BUILD_DIR}" ]]; then
    echo "Error: Host build directory does not exist: ${HOST_BUILD_DIR}" >&2
    echo "Run scripts/configure_build.sh first." >&2
    exit 1
fi

echo "Host build OK: ${HOST_BUILD_DIR}"

CMAKE_TARGETS=(
    # Torq compiler tool (the main binary)
    torq-compile
    # Torq compiler Python bindings (symlinks Python source into build tree)
    torq_compiler_python_bindings
)

for target in "${CMAKE_TARGETS[@]}"; do
    echo "Building target: ${target}"
    cmake --build "${HOST_BUILD_DIR}" --target "${target}"
done

mkdir -p "${OUTPUT_DIR}"
export TORQ_COMPILER_CMAKE_BUILD_DIR="${HOST_BUILD_DIR}"

if [[ -z "${TORQ_WHEEL_VERSION:-}" ]]; then
    TIMESTAMP="$(date -u '+%Y%m%dT%H%M%SZ')"
    COMMIT="$(cd "${PROJECT_DIR}" && git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
    export TORQ_WHEEL_VERSION="0.dev0+${TIMESTAMP}.g${COMMIT}"
fi
echo "Using version: ${TORQ_WHEEL_VERSION}"

pip wheel \
    --no-build-isolation \
    --no-deps \
    -w "${OUTPUT_DIR}" \
    "${PROJECT_DIR}/compiler/"

WHEEL_FILE="$(ls -t "${OUTPUT_DIR}"/torq_compiler-*.whl 2>/dev/null | head -1)"
echo ""
echo "Done. Wheel: ${WHEEL_FILE}"
