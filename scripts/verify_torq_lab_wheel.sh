#!/bin/bash
# Verify that torq.lab is packaged in a built torq-compiler wheel.
#
# Installs the wheel(s) into a throwaway venv (with no checkout .pth and no
# system site packages) and asserts that torq.lab imports from the installed
# package, that the CLI entry points work, and that the wheel archive contains
# the torq.lab modules. This is the authoritative packaging check: it fails the
# same way the latent torq.gen_config gap would, catching code that only works
# via the checkout python/ path.
#
# Usage:
#   scripts/verify_torq_lab_wheel.sh [compiler-wheel] [extra-wheel ...]
#
# With no arguments it uses the newest dist/torq_compiler-*.whl.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -ge 1 ]]; then
    COMPILER_WHEEL="$1"; shift
    EXTRA_WHEELS=("$@")
else
    COMPILER_WHEEL="$(ls -t "${PROJECT_DIR}"/dist/torq_compiler-*.whl 2>/dev/null | head -1 || true)"
    EXTRA_WHEELS=()
fi

if [[ -z "${COMPILER_WHEEL}" || ! -f "${COMPILER_WHEEL}" ]]; then
    echo "Error: compiler wheel not found: ${COMPILER_WHEEL:-<none>}" >&2
    echo "Build it first: scripts/build_compiler_wheel.sh <host-build-dir>" >&2
    exit 1
fi

echo "Compiler wheel: ${COMPILER_WHEEL}"

echo "=== wheel contains torq/lab modules ==="
for f in torq/lab/__init__.py torq/lab/profiling.py torq/lab/model_profiler/perfetto_logger.py; do
    if ! unzip -l "${COMPILER_WHEEL}" | grep -q "${f}"; then
        echo "Error: ${f} is not in the wheel." >&2
        exit 1
    fi
done
unzip -l "${COMPILER_WHEEL}" | grep "torq/lab/" || true

VENV="$(mktemp -d)/venv"
cleanup() { rm -rf "$(dirname "${VENV}")"; }
trap cleanup EXIT

echo "=== creating fresh venv (no system site packages, no checkout .pth) ==="
python3 -m venv "${VENV}"
"${VENV}/bin/pip" install -q --upgrade pip
"${VENV}/bin/pip" install -q "${COMPILER_WHEEL}" "${EXTRA_WHEELS[@]}"

echo "=== import resolves from the installed package, not the checkout ==="
"${VENV}/bin/python" - <<'PY'
import torq.lab
# torq.lab.remote pulls in torq.lab.transport; importing it proves the package is
# self-contained in the wheel (no dependency on the checkout torq.utils tree).
import torq.lab.remote  # noqa: F401
# torq.lab.profiling must import without the optional [profile] extra installed.
import torq.lab.profiling  # noqa: F401
path = torq.lab.__file__
print("torq.lab:", path)
assert "site-packages" in path, f"torq.lab did not import from site-packages: {path}"
assert "/src/python/" not in path, f"torq.lab imported from the checkout: {path}"
PY

echo "=== python -m torq.lab --help ==="
"${VENV}/bin/python" -m torq.lab --help >/dev/null

echo "=== torq-lab --help (console script) ==="
"${VENV}/bin/torq-lab" --help >/dev/null

echo "=== [profile] extra: install and import the profiling cluster ==="
"${VENV}/bin/pip" install -q "${COMPILER_WHEEL}[profile]" "${EXTRA_WHEELS[@]}"
"${VENV}/bin/python" - <<'PY'
import torq.lab.profiling as profiling
import torq.lab.model_profiler.perfetto_logger as pl
assert profiling.pd is not None, "profiling deps not active after installing [profile]"
for mod in (profiling, pl):
    assert "site-packages" in mod.__file__, f"{mod.__name__} not from site-packages: {mod.__file__}"
    assert "/src/python/" not in mod.__file__, f"{mod.__name__} imported from the checkout: {mod.__file__}"
print("torq.lab.profiling + model_profiler.perfetto_logger OK from [profile] extra")
PY

echo "OK: torq.lab is packaged and importable from the wheel."
