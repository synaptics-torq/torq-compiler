"""Regression test for synaptics-torq/torq-compiler-dev#2351.

Probe diagnostic handlers live on the shared MLIRContext while sibling
functions probe concurrently, so a handler could latch another thread's
diagnostic. This is a race: one passing run proves nothing, so the module is
compiled REPEATS times and every run must succeed. A failure here is a real
regression, not flakiness -- find which handler site regressed rather than
marking this flaky. Raise TORQ_PROBE_RACE_REPEATS for a stronger local check.
"""

import os
import subprocess

import pytest

from torq.testing.iree import MODELS_DIR

REPEATS = int(os.environ.get("TORQ_PROBE_RACE_REPEATS", "10"))

MODEL = MODELS_DIR / "tiling" / "matmul-two-functions-bf16.mlir"

FLAGS = [
    "--torq-hw=SL2610",
    "--torq-convert-dtypes",
    "--torq-convert-io-dtype",
    "--torq-disable-slicing",
    "--torq-enable-split-constants-optimization",
    "--torq-enable-annotate-tied-operands",
    "--torq-max-nss-programs-size", "0x939E00",
    "--torq-disable-css",
    "--torq-disable-host",
]

# A release build reports the race as an LRAM or translation failure instead of
# tripping the assert, so the return code alone is not enough.
RACE_SIGNATURES = (
    "Assertion",
    "failed to allocate LRAM addresses",
    "failed to run translation of source executable to target executable",
)


@pytest.mark.ci
def test_probe_diag_handler_is_thread_safe(torq_compiler, tmp_path):
    for i in range(REPEATS):
        vmfb = tmp_path / f"out-{i}.vmfb"
        cmd = [str(torq_compiler.file_path), str(MODEL), "-o", str(vmfb), *FLAGS]
        proc = subprocess.run(cmd, capture_output=True, text=True)

        tail = "\n".join(proc.stderr.splitlines()[-40:])
        assert proc.returncode == 0, (
            f"run {i}/{REPEATS} failed with returncode {proc.returncode}:\n{tail}"
        )
        for signature in RACE_SIGNATURES:
            assert signature not in proc.stderr, (
                f"run {i}/{REPEATS} hit race signature {signature!r}:\n{tail}"
            )
        assert vmfb.stat().st_size > 0, f"run {i}/{REPEATS} produced an empty binary"


@pytest.mark.ci
def test_probe_diag_input_compiles_single_threaded(torq_compiler, tmp_path):
    """Control: separates the race from the input simply not fitting."""
    vmfb = tmp_path / "out-single.vmfb"
    cmd = [
        str(torq_compiler.file_path), str(MODEL), "-o", str(vmfb),
        *FLAGS, "--mlir-disable-threading",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 0, f"single-threaded control failed:\n{proc.stderr}"
    assert vmfb.stat().st_size > 0, "single-threaded control produced an empty binary"
