import re
import subprocess
from pathlib import Path

import pytest

from torq.testing.iree import MODELS_DIR


def _phase_named(phases_dir, name):
    """Return the text of the compilation-phase dump whose file name is
    `<model>.<n>.<name>.mlir` (the highest-numbered one), or None."""
    files = sorted(phases_dir.glob(f"*.{name}.mlir"))
    if not files:
        return None
    return files[-1].read_text()


@pytest.mark.ci
def test_sliced_per_channel_rescale_pinned(torq_compiler, tmp_path):
    """Pin the sliced-table lowering of the per-channel rescale (RescaleOpConversion).

    With C=2048, LinalgSlicing splits the elementwise chain along the channel dim on
    SL2610, so the rescale's per-channel tables arrive as IV-dependent
    tensor.extract_slices. The lowering must take the sliced-table path: the FMA reads
    its {bias, scale} rows through a windowed slice of the normalized [2048, 2] table.
    If the slicing heuristics ever stop splitting C=2048, the plain fold path is used
    instead (the FMA DMAs the whole table, unwindowed) and this test fails loudly.
    """
    model = MODELS_DIR / "linalg_ops" / "rescale-per-channel-sliced-i32-i8.mlir"
    phases = tmp_path / "phases"

    cmd = [
        str(torq_compiler.file_path),
        str(model),
        "-o",
        str(tmp_path / "model.vmfb"),
        "--torq-hw=SL2610",
        "--torq-target-host-triple=native",
        "--iree-input-type=linalg-torq",
        "--torq-disable-host",
        "--torq-disable-css",
        f"--dump-compilation-phases-to={phases}",
    ]
    print("Compiling with:", " ".join(cmd))
    subprocess.check_call(cmd)

    ir = _phase_named(phases, "executable-targets")
    assert ir, "no executable-targets phase was dumped"

    # The rescale must have lowered to an FMA program (not CSS/host loops).
    assert "slice_program_torq_hl.fma" in ir, (
        "no torq_hl.fma slice program — the per-channel rescale did not lower to the "
        "FMA (sliced-table path taken?)"
    )

    # The normalized whole table: one [2048, 2] i32 constant, deduplicated across the
    # slice programs.
    whole_table = re.findall(r"memref\.global \"private\" constant @\w+ : memref<2048x2xi32>", ir)
    assert len(whole_table) == 1, (
        f"expected exactly one deduplicated [2048, 2] normalized table constant, "
        f"found {len(whole_table)}"
    )

    # The FMA programs read a strict window of it (a subview of the [2048, 2] table
    # with a smaller first dim), mirroring the data slice.
    window = re.search(
        r"memref\.subview %\w+\[\d+, 0\] \[(\d+), 2\] \[1, 1\][^\n]*: "
        r"memref<2048x2xi32>",
        ir,
    )
    assert window, (
        "no windowed subview of the [2048, 2] table — the rescale took the plain fold "
        "path (is LinalgSlicing still splitting C=2048 on SL2610?)"
    )
    assert int(window.group(1)) < 2048, (
        f"the table subview is not windowed (first dim {window.group(1)})"
    )
