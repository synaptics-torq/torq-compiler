"""Compile-only regression tests.

Every .mlir in tests/testdata/linalg_compilation/ must compile cleanly for
each chip in the active test plan; the compiled module is not run. This
suite exists for patterns whose numerics cannot be validated on the
available reference paths (each case header explains why) while still
guarding the compilation itself — dispatch formation, tiling, NDL bounds
checks — against regressions.

Per-case compiler options are declared in the case file itself with a
`// TORQ_COMPILE_OPTIONS: --flag1 --flag2` comment, so each case stays
self-contained. Like the linalg_ops suite, cases are forced onto the
NSS/slice path so a change that silently routes them to the host/CSS
fallback fails loudly.

The suite is `ci`-marked only: the cases are not run, so the fpga_ci lane
would just recompile them for the same chip group without any runtime
coverage.
"""

import pytest

from torq.testing.cases import get_test_cases_from_files
from torq.testing.iree import list_mlir_files


def _compile_options_from_mlir(mlir_file):
    """Per-case compiler options declared in the file as
    `// TORQ_COMPILE_OPTIONS: --flag1 --flag2` (same comment-directive style
    as the TORQ_FP_MAX_TOL comparison directives)."""
    options = []
    for line in mlir_file.read_text().splitlines():
        if "TORQ_COMPILE_OPTIONS:" not in line:
            continue
        options += line.split("TORQ_COMPILE_OPTIONS:", 1)[1].split()
    return options


# Force every case onto the NSS/slice path (see test_linalg_ops.py).
NSS_ONLY = ["--torq-disable-host", "--torq-disable-css"]


@pytest.fixture(params=get_test_cases_from_files(
    list_mlir_files("linalg_compilation")
))
def case_config(request, runtime_hw_type, chip_config):
    return {
        "mlir_model_file": "static_mlir_model_file",
        "static_mlir_model_file": request.param.data,
        "torq_compiler_options": _compile_options_from_mlir(request.param.data) + NSS_ONLY,
    }


@pytest.mark.ci
def test_linalg_compilation(torq_compiled_model):
    # Reaching here means the case compiled for the active chip; the fixture
    # aborts the test on any compilation failure.
    assert torq_compiled_model.data.exists()
