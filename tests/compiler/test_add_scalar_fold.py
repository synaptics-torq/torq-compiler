import subprocess

import pytest

from torq.testing.iree import MODELS_DIR

# Int16 tensor minus a rescaled scalar constant, against a 1x6x8x4 output.
RESCALED_SCALAR_MODEL = MODELS_DIR / "tosa_ops" / "sub-rescaled-const-scalar-i16.mlir"


@pytest.mark.ci
def test_rescaled_scalar_constant_folds_into_the_bias(torq_compiler, tmp_path):
    """The fused add must fold a rescaled scalar constant into its bias. A broadcast of the
    constant means that the alignment reached it first, and the add uses the ALU for it."""
    cmd = [
        str(torq_compiler.file_path),
        str(RESCALED_SCALAR_MODEL),
        "-o",
        str(tmp_path / "model.vmfb"),
        "--torq-hw=SL2610",
        "--torq-target-host-triple=native",
        "--torq-disable-host",
        "--torq-disable-css",
        "--mlir-disable-threading",
        "--mlir-print-ir-after=torq-fold-convert",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"compile failed:\n{proc.stderr}"
    # Without this, the test would pass even if the dump stopped showing the add.
    assert "torq_hl.add" in proc.stderr, "no torq_hl.add in the dump, so this test proves nothing"
    assert "torq_hl.broadcast" not in proc.stderr, "the scalar constant was broadcast"
