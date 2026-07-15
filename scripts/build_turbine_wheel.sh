#!/bin/bash
# Build a Python wheel for the third_party/iree-turbine (torq-turbine) submodule.
#
# Usage:
#   $0
#
# The finished wheel is placed in dist/.
#
# Environment variables:
#   TORQ_WHEEL_VERSION  Optional version string. When unset the wheel uses the
#                       default dev version from the submodule setup.py.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_DIR="${PROJECT_DIR}/dist"
TURBINE_DIR="${PROJECT_DIR}/third_party/iree-turbine"

if [[ ! -f "${TURBINE_DIR}/setup.py" ]]; then
    echo "Error: torq-turbine source not found at ${TURBINE_DIR}" >&2
    echo "Run: git submodule update --init third_party/iree-turbine" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"

# Export TORQ_WHEEL_VERSION so the submodule setup.py can align the wheel
# version with the compiler/runtime wheels when the caller sets it.
export TORQ_WHEEL_VERSION="${TORQ_WHEEL_VERSION:-}"

pip wheel \
    --no-build-isolation \
    --no-deps \
    -w "${OUTPUT_DIR}" \
    "${TURBINE_DIR}"

WHEEL_FILE="$(ls -t "${OUTPUT_DIR}"/torq_turbine-*.whl 2>/dev/null | head -1)"
echo ""
echo "Done. Wheel: ${WHEEL_FILE}"
