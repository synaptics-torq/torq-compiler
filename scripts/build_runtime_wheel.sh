#!/bin/bash
# Build a torq-runtime Python wheel.
#
# Usage:
#   $0 <host-build-dir> [soc-build-dir] [target] [toolchain]
#
# Host build:
#   scripts/build_runtime_wheel.sh ../iree-build
#
# Cross-compile for SoC:
#   scripts/build_runtime_wheel.sh ../iree-build ../iree-build-soc-python astra_machina poky
#
# The finished wheel is placed in dist/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${PROJECT_DIR}/dist"

function usage () {
    echo "$0 <host-build-dir> [soc-build-dir] [target] [toolchain]"
}

if [[ -z "${1:-}" ]]; then
    echo "Error: host-build-dir is required." >&2
    usage
    exit 1
fi

HOST_BUILD_DIR="$(cd "${PROJECT_DIR}" && readlink -f "$1")"
SOC_BUILD_DIR="${2:-}"
TARGET="${3:-}"
TOOLCHAIN="${4:-}"

CROSS_COMPILE=false
if [[ -n "${TARGET}" ]]; then
    CROSS_COMPILE=true
    SOC_BUILD_DIR="${SOC_BUILD_DIR:-${HOST_BUILD_DIR}-soc-python}"
    SOC_BUILD_DIR="$(cd "${PROJECT_DIR}" && mkdir -p "${SOC_BUILD_DIR}" && readlink -f "${SOC_BUILD_DIR}")"
fi

if [[ ! -d "${HOST_BUILD_DIR}" ]]; then
    echo "Error: Host build directory does not exist: ${HOST_BUILD_DIR}" >&2
    echo "Run scripts/configure_build.sh first." >&2
    exit 1
fi

HOST_TOOLS_DIR="${HOST_BUILD_DIR}/third_party/iree/tools"
if [[ ! -d "${HOST_TOOLS_DIR}" ]]; then
    echo "Error: Host tools not found at ${HOST_TOOLS_DIR}" >&2
    echo "Ensure the host build has been built at least once." >&2
    exit 1
fi
echo "Host build OK: ${HOST_BUILD_DIR}"

CMAKE_BUILD_DIR=""
if [[ "${CROSS_COMPILE}" == true ]]; then
    echo "Configuring cross-compile build (target=${TARGET}, toolchain=${TOOLCHAIN:-default})"

    CONFIGURE_ARGS=("${SOC_BUILD_DIR}" "${HOST_BUILD_DIR}" "${TARGET}")
    if [[ -n "${TOOLCHAIN}" ]]; then
        CONFIGURE_ARGS+=("${TOOLCHAIN}")
    fi
    "${SCRIPT_DIR}/configure_soc_build.sh" "${CONFIGURE_ARGS[@]}"

    CMAKE_BUILD_DIR="${SOC_BUILD_DIR}"
else
    echo "Building for native host"
    CMAKE_BUILD_DIR="${HOST_BUILD_DIR}"
fi

CMAKE_TARGETS=(
    # Defined by iree_py_library(NAME runtime) in
    # third_party/iree/runtime/bindings/python/CMakeLists.txt.
    # Also transitively builds all IREE CLI tools (iree-run-module, etc.).
    iree_runtime_bindings_python_runtime
    torq_runtime_python_bindings
    torq-run-module
)

for target in "${CMAKE_TARGETS[@]}"; do
    echo "Building target: ${target}"
    cmake --build "${CMAKE_BUILD_DIR}" --target "${target}"
done

mkdir -p "${OUTPUT_DIR}"
export TORQ_RUNTIME_CMAKE_BUILD_DIR="${CMAKE_BUILD_DIR}"

# Compute version once and export it so setup.py uses the same value across all invocations
if [[ -z "${TORQ_WHEEL_VERSION:-}" ]]; then
    TIMESTAMP="$(date -u '+%Y%m%dT%H%M%SZ')"
    COMMIT="$(cd "${PROJECT_DIR}" && git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
    export TORQ_WHEEL_VERSION="0.dev0+${TIMESTAMP}.g${COMMIT}"
    echo "Using version: ${TORQ_WHEEL_VERSION}"
fi

pip wheel \
    --no-build-isolation \
    --no-deps \
    -w "${OUTPUT_DIR}" \
    "${PROJECT_DIR}/runtime/"

WHEEL_FILE="$(ls -t "${OUTPUT_DIR}"/torq_runtime-*.whl 2>/dev/null | head -1)"
echo ""
echo "Done. Wheel: ${WHEEL_FILE}"
