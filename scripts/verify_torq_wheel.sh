#!/bin/bash
# Verify wheel packaging of the pure-Python torq libraries.
#
# Installs the wheel(s) into a throwaway venv (with no checkout .pth and no
# system site packages) and asserts:
#
#   torq.lab        — packaged, imports from site-packages, `torq-lab` CLI works
#                     (with and without the [profile] extra)
#   torq.gen_config — packaged, `torq-gen-config` CLI works (with the [onnx]
#                     extra, which the CLI import chain requires), and imports
#                     do not pull in pytest
#   exclusions      — torq.testing and torq.utils must NOT be in the wheel:
#                     they are the in-tree test framework and board/plot
#                     utilities, which the standalone gen_config path never
#                     imports, and no packaged torq/gen_config module may
#                     import pytest
#
# This is the authoritative packaging check: it fails the same way a missing
# package or stale entry point would at a user's site, catching code that only
# works via the checkout python/ path.
#
# Usage:
#   scripts/verify_torq_wheel.sh [compiler-wheel] [extra-wheel ...]
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

echo "=== wheel contains torq/lab and torq/gen_config modules ==="
for f in \
    torq/lab/__init__.py \
    torq/lab/profiling/annotate.py \
    torq/lab/profiling/perfetto.py \
    torq/lab/model_tools/extraction/onnx/layers.py \
    torq/lab/model_tools/dtype_conversion/onnx.py \
    torq/lab/logging.py \
    torq/lab/metrics.py \
    torq/lab/model_tools/extraction/tflite/layers.py \
    torq/lab/model_tools/shape_conversion/__init__.py \
    torq/lab/model_tools/shape_conversion/tflite.py \
    torq/lab/quantization/onnx/cli.py \
    torq/lab/quantization/onnx/dynamic/__init__.py \
    torq/lab/quantization/onnx/dynamic/_analysis.py \
    torq/lab/quantization/onnx/dynamic/_quantization.py \
    torq/lab/quantization/onnx/static/__init__.py \
    torq/lab/quantization/onnx/static/_quantization.py \
    torq/lab/quantization/onnx/weights/__init__.py \
    torq/lab/quantization/onnx/weights/_analysis.py \
    torq/lab/quantization/onnx/weights/_config.py \
    torq/lab/quantization/onnx/weights/_quantization.py \
    torq/lab/reporting.py \
    torq/lab/utils/cli.py \
    torq/lab/utils/onnxruntime.py \
    torq/gen_config/__init__.py \
    torq/gen_config/cli.py \
    torq/gen_config/_runner.py \
    torq/gen_config/_options.py \
    torq/gen_config/_cache.py \
; do
    if ! unzip -Z1 "${COMPILER_WHEEL}" "${f}" >/dev/null 2>&1; then
        echo "Error: ${f} is not in the wheel." >&2
        exit 1
    fi
done

echo "=== wheel must NOT contain torq/testing or torq/utils ==="
if unzip -Z1 "${COMPILER_WHEEL}" | grep -E "torq/(testing|utils)/" >/dev/null; then
    echo "Error: torq.testing / torq.utils leaked into the wheel:" >&2
    unzip -l "${COMPILER_WHEEL}" | grep -E "torq/(testing|utils)/" >&2
    exit 1
fi

echo "=== no torq/gen_config module in the wheel imports pytest ==="
while IFS= read -r f; do
    if unzip -p "${COMPILER_WHEEL}" "${f}" | grep -E "^[[:space:]]*(import pytest|from pytest)" >/dev/null; then
        echo "Error: ${f} in the wheel imports pytest:" >&2
        unzip -p "${COMPILER_WHEEL}" "${f}" | grep -nE "^[[:space:]]*(import pytest|from pytest)" >&2
        exit 1
    fi
done < <(unzip -Z1 "${COMPILER_WHEEL}" "torq/gen_config/*.py")

VENV="$(mktemp -d)/venv"
cleanup() { rm -rf "$(dirname "${VENV}")"; }
trap cleanup EXIT

echo "=== creating fresh venv (no system site packages, no checkout .pth) ==="
python3 -m venv "${VENV}"
"${VENV}/bin/pip" install -q --upgrade pip
"${VENV}/bin/pip" install -q "${COMPILER_WHEEL}" "${EXTRA_WHEELS[@]}"

echo "=== torq.lab imports resolve from the installed package, not the checkout ==="
"${VENV}/bin/python" - <<'PY'
import torq.lab
# torq.lab.pipeline.remote pulls in the SSH/ADB transport; importing it proves the
# package is self-contained in the wheel (no dependency on the checkout torq.utils
# tree).
import torq.lab.pipeline.remote  # noqa: F401
# torq.lab.profiling must import without the optional [profile] extra installed.
import torq.lab.profiling  # noqa: F401
import torq.lab.metrics as metrics
import torq.lab.reporting as reporting
path = torq.lab.__file__
print("torq.lab:", path)
assert "site-packages" in path, f"torq.lab did not import from site-packages: {path}"
assert "/src/python/" not in path, f"torq.lab imported from the checkout: {path}"
for mod in (metrics, reporting):
    assert "site-packages" in mod.__file__, f"{mod.__name__} not from site-packages: {mod.__file__}"
assert callable(metrics.measure_time)
assert callable(reporting.ReportGenerator)
PY

echo "=== python -m torq.lab --help ==="
"${VENV}/bin/python" -m torq.lab --help >/dev/null

echo "=== torq-lab --help (console script) ==="
"${VENV}/bin/torq-lab" --help >/dev/null

echo "=== [onnx] extra: install and exercise torq-gen-config ==="
"${VENV}/bin/pip" install -q "${COMPILER_WHEEL}[onnx]" "${EXTRA_WHEELS[@]}"
"${VENV}/bin/python" - <<'PY'
import sys

import torq.gen_config.cli as cli
import torq.gen_config._runner as runner  # noqa: F401
path = cli.__file__
print("torq.gen_config.cli:", path)
assert "site-packages" in path, f"torq.gen_config did not import from site-packages: {path}"
assert "/src/python/" not in path, f"torq.gen_config imported from the checkout: {path}"
assert "pytest" not in sys.modules, "importing torq.gen_config.cli pulled in pytest"

for absent in ("torq.testing", "torq.utils"):
    try:
        __import__(absent)
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError(f"{absent} must not be importable from the wheel")
print("torq.gen_config OK from [onnx] extra; no pytest; testing/utils absent")
PY

echo "=== torq-gen-config --help (console script) ==="
"${VENV}/bin/torq-gen-config" --help >/dev/null

echo "=== torq-convert-dtype --help (console script) ==="
"${VENV}/bin/torq-convert-dtype" --help >/dev/null
"${VENV}/bin/torq-convert-dtype" onnx --help >/dev/null

echo "=== torq-convert-static --help (console script) ==="
"${VENV}/bin/torq-convert-static" --help >/dev/null
"${VENV}/bin/torq-convert-static" tflite --help >/dev/null

echo "=== python -m torq.gen_config --help ==="
"${VENV}/bin/python" -m torq.gen_config --help >/dev/null

echo "=== [profile] extra: install and import the profiling cluster ==="
"${VENV}/bin/pip" install -q "${COMPILER_WHEEL}[profile]" "${EXTRA_WHEELS[@]}"
"${VENV}/bin/python" - <<'PY'
import torq.lab.profiling.annotate as annotate
import torq.lab.profiling.perfetto as pl
assert annotate.pd is not None, "profiling deps not active after installing [profile]"
for mod in (annotate, pl):
    assert "site-packages" in mod.__file__, f"{mod.__name__} not from site-packages: {mod.__file__}"
    assert "/src/python/" not in mod.__file__, f"{mod.__name__} imported from the checkout: {mod.__file__}"
print("torq.lab.profiling.annotate + profiling.perfetto OK from [profile] extra")
PY

echo "OK: torq.lab and torq.gen_config are packaged and importable from the wheel; torq.testing/torq.utils are excluded, and no packaged torq/gen_config module imports pytest."
