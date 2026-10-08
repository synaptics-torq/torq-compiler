import re
import subprocess

import pytest

from torq.testing.iree import MODELS_DIR


_CONSTANT = re.compile(r"(%[\w]+) = arith.constant (\d+) : index")
_FOR = re.compile(r"scf.for %\w+ = %\w+ to (%\w+) step (%\w+)")


def _tile_steps(torq_compiler, tmp_path, fixture):
    # Run the full pipeline. The single-pass setup in test_matmul_tiling.py
    # aborts on elementwise dispatches.
    result = subprocess.run(
        [
            str(torq_compiler.file_path),
            str(MODELS_DIR / fixture),
            "-o", str(tmp_path / "x.vmfb"),
            "--torq-hw=SL2610",
            "--torq-target-host-triple=native",
            "--torq-disable-css",
            "--torq-disable-host",
            "--mlir-print-ir-after=torq-tile-and-fuse",
            "--mlir-disable-threading",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr[-4000:]
    ir = result.stderr
    # Read the dump in order, so each loop uses the constants of its own
    # function. SSA names repeat across functions.
    constants = {}
    loops = []
    for line in ir.splitlines():
        for name, value in _CONSTANT.findall(line):
            constants[name] = int(value)
        for upper, step in _FOR.findall(line):
            loops.append((constants.get(upper), constants.get(step)))
    return loops, ir


@pytest.mark.ci
@pytest.mark.parametrize(
    "fixture, domain, step",
    [
        # Tile 39 without the rule. 39 is not a multiple of the chunk
        # (2 slices x 4 = 8), so 7 rows of each tile of 39 run on one slice.
        ("torch_ops/erf_mlp_dup_lram_load_bf16.mlir", 192, 32),
        # qdq-mul-broadcast.mlir with a constant per-channel scale, so
        # LinalgSlicing slices the group. Tile 11 without the rule.
        ("torch_ops/qdq-mul-broadcast-const-scale.mlir", 32, 8),
        # LinalgSlicing does not slice this group (the mul has two live inputs,
        # a residual add), so the tile is not rounded and stays at 11.
        ("torch_ops/qdq-mul-broadcast.mlir", 32, 11),
    ],
)
def test_slice_aligned_tile_step(torq_compiler, tmp_path, fixture, domain, step):
    loops, ir = _tile_steps(torq_compiler, tmp_path, fixture)
    steps = [s for upper, s in loops if upper == domain]
    if not steps:
        pytest.fail(f"no scf.for over {domain} after torq-tile-and-fuse:\n{ir[:2000]}")
    assert steps[0] == step
