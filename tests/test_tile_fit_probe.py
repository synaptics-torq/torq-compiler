import re
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


# On SR200, the largest tile of the block 1 group of this model fits only if the LRAM
# plan relocates buffers. A relocation stores them to XRAM and loads them straight back,
# so each relocated buffer gets an XRAM copy.
RELOCATION_MODEL = MODELS_DIR / "tiling" / "mbv2-block1-relocation-i8.mlir"
SR200_HW = "SR200"


def count_xram_copies(torq_compiler, tmp_path, extra_flags):
    """Count the XRAM allocations in the dispatch after LRAM address assignment."""
    cmd = [
        str(torq_compiler.file_path),
        str(RELOCATION_MODEL),
        "-o",
        str(tmp_path / "model.vmfb"),
        f"--torq-hw={SR200_HW}",
        "--torq-target-host-triple=native",
        "--torq-disable-host",
        "--torq-disable-css",
        "--mlir-disable-threading",
        "--mlir-elide-elementsattrs-if-larger=16",
        "--mlir-print-ir-after=torq-assign-lram-addresses",
    ] + extra_flags
    print("Compiling with:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"compile failed:\n{proc.stderr}"

    # The probe prints its own runs as @fit_tile_to_memory_*; keep the dispatch.
    dumps = [
        dump
        for dump in proc.stderr.split("// -----// IR Dump After")
        if re.search(r"func\.func @main_dispatch", dump)
    ]
    assert dumps, "no dump of the dispatch after torq-assign-lram-addresses"
    return len(re.findall(r"memref\.alloc\(\) : memref<[^>]*mem_space = xram", dumps[-1]))


@pytest.mark.ci
def test_probe_avoids_cheap_relocation(torq_compiler, tmp_path):
    """The probe must prefer one more tile to a tile that fits only with a relocation."""
    relocated = count_xram_copies(
        torq_compiler, tmp_path, ["--torq-probe-avoid-relocation=false"]
    )
    # Without this, the test would pass even if the model stopped relocating.
    assert relocated > 0, "the model no longer relocates, so this test proves nothing"

    # Compare with the count above: an unrelated XRAM buffer must not fail the test.
    assert count_xram_copies(torq_compiler, tmp_path, []) < relocated


@pytest.mark.ci
def test_probe_keeps_relocation_without_tile_budget(torq_compiler, tmp_path):
    """Without a budget for more tiles, the probe must keep the tile that relocates."""
    relocated = count_xram_copies(
        torq_compiler, tmp_path, ["--torq-probe-avoid-relocation=false"]
    )
    assert relocated > 0, "the model no longer relocates, so this test proves nothing"

    no_budget = count_xram_copies(
        torq_compiler, tmp_path, ["--torq-probe-relocation-max-tile-growth=0"]
    )
    assert no_budget == relocated

    # A relocation smaller than the minimum is not worth an extra tile.
    below_minimum = count_xram_copies(
        torq_compiler, tmp_path, [f"--torq-probe-relocation-min-bytes={1 << 30}"]
    )
    assert below_minimum == relocated


def tile_loop_step(torq_compiler, tmp_path, model, hw, extra_flags, extent):
    """Compile `model` and return the step of the last tiled loop over `extent`.

    The last loop is the group of the model output. A producer group can also loop over
    `extent` before it.
    """
    cmd = [
        str(torq_compiler.file_path),
        str(model),
        "-o",
        str(tmp_path / "model.vmfb"),
        f"--torq-hw={hw}",
        "--torq-target-host-triple=native",
        "--mlir-disable-threading",
        "--mlir-elide-elementsattrs-if-larger=16",
        "--mlir-print-ir-after=torq-tile-and-fuse",
    ] + extra_flags
    print("Compiling with:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"compile failed:\n{proc.stderr}"

    steps = re.findall(rf"scf\.for %\S+ = %\S+ to %c{extent}(?:_\d+)? step %c(\d+)", proc.stderr)
    assert steps, f"no tiled loop over {extent}"
    return int(steps[-1])


# The resnet50 conv5_block2_1_conv layer: 4 tiles of 128 output channels relocate a buffer
# on SL2610, and the only relocation-free tile within the budget has 103 channels.
CHANNEL_MODEL = MODELS_DIR / "tiling" / "resnet50-conv5-block2-1-conv-i8.mlir"


@pytest.mark.ci
def test_relocation_free_search_keeps_output_channels(torq_compiler, tmp_path):
    """The search must not cut the output channels of the principal op to avoid a relocation."""
    flags = ["--torq-disable-host", "--torq-disable-css"]
    default = tile_loop_step(torq_compiler, tmp_path, CHANNEL_MODEL, "SL2610", flags, 512)
    disabled = tile_loop_step(
        torq_compiler,
        tmp_path,
        CHANNEL_MODEL,
        "SL2610",
        flags + ["--torq-probe-avoid-relocation=false"],
        512,
    )
    assert default == disabled


# encoder_bf16_layer_Conv_2 on SR200: 7 tiles of 192 columns relocate a buffer. The
# smallest relocation-free tile has 157 columns, and 160 also avoids the relocation.
ALIGNMENT_MODEL = MODELS_DIR / "torch_ops" / "encoder_bf16_layer_Conv_2-sim-sl2610-v1.mlir"


@pytest.mark.ci
def test_relocation_free_search_keeps_alignment(torq_compiler, tmp_path):
    """The relocation-free tile must keep the alignment of the original tile, up to 4."""
    disabled = tile_loop_step(
        torq_compiler,
        tmp_path,
        ALIGNMENT_MODEL,
        SR200_HW,
        ["--torq-probe-avoid-relocation=false"],
        1249,
    )
    default = tile_loop_step(torq_compiler, tmp_path, ALIGNMENT_MODEL, SR200_HW, [], 1249)
    # Without this, the test would pass even if the search stopped changing the tile.
    assert default != disabled, "the tile no longer relocates, so this test proves nothing"
    assert default % 4 == 0, f"tile of {default} columns is not a multiple of 4"


# Tile-and-fuse must not fuse a producer that a tiled loop would recompute on every
# iteration, and must keep fusing it when nothing would be recomputed. Both cases feed a
# matmul's LHS from an elementwise producer (arith.negf).
PRODUCER_N_TILED_MODEL = MODELS_DIR / "tiling" / "matmul-lhs-producer-n-tiled-bf16.mlir"
PRODUCER_M_TILED_MODEL = MODELS_DIR / "tiling" / "matmul-lhs-producer-m-tiled-bf16.mlir"


def outer_loop_nests(torq_compiler, tmp_path, model, extra_flags):
    """For each outermost scf.for nest of the dispatch after tile-and-fuse, whether it
    contains the producer (arith.negf) and the matmul."""
    cmd = [
        str(torq_compiler.file_path),
        str(model),
        "-o",
        str(tmp_path / "model.vmfb"),
        "--torq-hw=SL2610",
        "--torq-disable-host",
        "--torq-disable-css",
        "--mlir-disable-threading",
        "--mlir-print-ir-after=torq-tile-and-fuse",
    ] + extra_flags
    print("Compiling with:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"compile failed:\n{proc.stderr}"

    dumps = [
        dump
        for dump in proc.stderr.split("// -----// IR Dump After")
        if re.search(r"func\.func @main_dispatch", dump) and "linalg.matmul" in dump
    ]
    assert dumps, "no dump of the matmul dispatch after torq-tile-and-fuse"

    nests, stack = [], []
    for line in dumps[-1].splitlines():
        text = line.strip()
        if text.startswith("}") and stack:
            stack.pop()
        if "scf.for" in stack:
            nests[-1]["negf"] |= "arith.negf" in text
            nests[-1]["matmul"] |= "linalg.matmul" in text
        if text.endswith("{"):
            if "scf.for" in text and "scf.for" not in stack:
                nests.append({"negf": False, "matmul": False})
            stack.append("scf.for" if "scf.for" in text else "other")
    return nests


def producer_fused(nests):
    return any(nest["negf"] and nest["matmul"] for nest in nests)


@pytest.mark.ci
def test_producer_not_recomputed_in_n_tiled_matmul(torq_compiler, tmp_path):
    """The LHS slice does not index N: fusing the producer would recompute it per N tile."""
    hatch = outer_loop_nests(
        torq_compiler,
        tmp_path,
        PRODUCER_N_TILED_MODEL,
        ["--torq-disable-tnf-avoid-producer-recompute"],
    )
    # Without this, the test would pass even if the matmul stopped being tiled along N.
    assert producer_fused(hatch), "control: the producer is not fused even without the check"

    nests = outer_loop_nests(torq_compiler, tmp_path, PRODUCER_N_TILED_MODEL, [])
    assert not producer_fused(nests), f"the producer is recomputed per N tile: {nests}"
    assert any(nest["negf"] and not nest["matmul"] for nest in nests), (
        f"the producer is not computed once in its own loop: {nests}"
    )


@pytest.mark.ci
def test_producer_fused_in_m_tiled_matmul(torq_compiler, tmp_path):
    """Near miss: tiled along M only, the LHS slice follows the tile, so nothing is
    recomputed and the producer must stay fused."""
    nests = outer_loop_nests(torq_compiler, tmp_path, PRODUCER_M_TILED_MODEL, [])
    assert any(nest["matmul"] for nest in nests), f"the matmul is no longer tiled: {nests}"
    assert producer_fused(nests), f"the producer is not fused although nothing is recomputed: {nests}"
