import subprocess

import pytest

from torq.testing.iree import MODELS_DIR

# The tile-fit probe is stricter than the real pipeline, so it can answer "no
# tile fits" for a block the real pipeline places fine. Reaching that answer
# needs an LRAM much smaller than any real chip: 8KB is 1/64 of an SL2610.
# `hw_id` is a number, not a name (SL2610 is 0).
TIGHT_LRAM_HW = "0:8:2:coral_v1:nss_v1"

# These compile at 8KB only if fitTileToMemory keeps the minimum tile instead of
# reporting a failure the caller reads as "this op cannot be tiled"
# (synaptics-torq/torq-compiler-dev#2305). The pooling cases show that the
# symptom is not specific to convolution.
NO_FIT_MODELS = [
    "linalg_ops/conv1d-ncw-fcw-multichannel-kw3-bias-bf16",
    "linalg_ops/conv1d-ncw-fcw-multichannel-kw3-f32out-bf16",
    "linalg_ops/conv1d-ncw-fcw-multichannel-kw7-bf16",
    "linalg_ops/conv1d-ncw-fcw-nnnr3-convtranspose-im2col-bf16",
    "linalg_ops/conv1d-ncw-fcw-pointwise-bias-bf16",
    "tosa_ops/maxpool2d-stride1-k3x3-32x32x4",
    "tosa_ops/maxpool2d-stride2-k3x3-mixed-phase-pad-32x32x8",
    "tosa_ops/maxpool_int16_batch1_3x3_stride1x1_16x4",
]


@pytest.mark.parametrize("model_name", NO_FIT_MODELS)
def test_no_probe_tile_fits_uses_minimum_tile(model_name, torq_compiler, tmp_path):
    """When no tile fits the memory probe, tile-and-fuse must use the minimum
    tile and let the real pipeline decide, not leave the op untiled.

    An untiled op is never smaller than the minimum tile, so leaving it untiled
    only moves the failure to the real allocation. Before the fix these eight
    models fail at 8KB LRAM with "operation can't be tiled: no more domains to
    tile" followed by "cannot allocate ...".
    """
    model = MODELS_DIR / f"{model_name}.mlir"
    vmfb = tmp_path / f"{model.stem}.vmfb"

    cmd = [
        str(torq_compiler.file_path),
        str(model),
        "-o",
        str(vmfb),
        f"--torq-hw={TIGHT_LRAM_HW}",
        "--torq-target-host-triple=native",
        "--torq-disable-host",
        "--torq-disable-css",
    ]
    print("Compiling with:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)

    assert proc.returncode == 0, f"compile failed:\n{proc.stderr}"
    assert vmfb.stat().st_size > 0, "compile produced an empty binary"

    # The point of the test is this path, so fail loudly if a later change stops
    # reaching it -- the compile would still pass and the test would be empty.
    assert "no tile fits the memory probe" in proc.stderr, (
        f"the no-fit path was never taken, so this test proves nothing:\n{proc.stderr}"
    )
