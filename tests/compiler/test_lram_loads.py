import re
import subprocess

import pytest

from torq.testing.iree import MODELS_DIR

from ..ops.keras_ops_models import (  # noqa: F401 (fixtures)
    keras_ops_case_data,
    keras_ops_cases,
    keras_ops_model,
)
from ..ops.torch_ops_models import torch_ops_mlir

# The tanh GELU of gelu-gemma.mlir. On SL2610 it runs untiled on both slices, and its LRAM plan
# spills copies of the input along with computed buffers.
GELU_MODEL = "gelu-gemma.mlir"

# Blocks 1 and 2 of MobileNetV2 (tests/testdata/keras_ops/mbv2_blocks.py). On SL2610 block 1
# builds the residual in a tile loop with peeled tiles. Block 2 builds its result in unrolled
# tiles, each one into the result of the last.
MBV2_BLOCKS12_CASE = "mbv2_blocks_1_2_112x112x16"

# A matmul that SL2610 splits on rows over the two slices, with a tail tile of 2 rows. The tail
# tile reads its init from the result of the 56 rows before it.
MATMUL_TAIL_MODEL = MODELS_DIR / "tosa_ops" / "matmul-in-int8-out-int32-noalign.mlir"

# A torq_hl.convert: the type of its output, then the type of its input.
CONVERT_RE = re.compile(
    r"= torq_hl\.convert out\(%\S+ : (tensor<.*?>)\) in\(%\S+ : (tensor<.*?>)\)"
)


def compile_stderr(torq_compiler, tmp_path, model, hw, extra_flags):
    """Compile `model` and return what the compiler printed to stderr."""
    cmd = [
        str(torq_compiler.file_path),
        str(model),
        "-o",
        str(tmp_path / "model.vmfb"),
        f"--torq-hw={hw}",
        "--torq-target-host-triple=native",
        "--torq-disable-host",
        "--torq-disable-css",
        "--mlir-disable-threading",
        "--mlir-elide-elementsattrs-if-larger=16",
    ] + extra_flags
    print("Compiling with:", " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"compile failed:\n{proc.stderr}"
    return proc.stderr


def dispatches_after(torq_compiler, tmp_path, model, hw, pass_arg, extra_flags):
    """Compile `model` and return the IR of each of its dispatches after `pass_arg`."""
    stderr = compile_stderr(
        torq_compiler, tmp_path, model, hw, [f"--mlir-print-ir-after={pass_arg}"] + extra_flags
    )

    # The tile-fit probe prints its own runs too. The last dump of a dispatch is the real one.
    dumps = {}
    for dump in stderr.split("// -----// IR Dump After"):
        name = re.search(r"func\.func @(\S+_dispatch\S*)\(", dump)
        if name:
            dumps[name.group(1)] = dump
    assert dumps, f"no dump of a dispatch after {pass_arg}"
    return list(dumps.values())


def dispatch_after(torq_compiler, tmp_path, model, hw, pass_arg, extra_flags):
    """Compile `model` and return the IR of its last dispatch after `pass_arg`."""
    return dispatches_after(torq_compiler, tmp_path, model, hw, pass_arg, extra_flags)[-1]


@pytest.fixture(params=[p for p in keras_ops_cases() if p.values[0].name == MBV2_BLOCKS12_CASE])
def case_config(request):
    """The keras_ops case of MBV2_BLOCKS12_CASE, for the tflite_mlir_model_file fixture."""
    data = request.param.data
    return {
        "_case": data,
        "keras_model": "keras_ops_model",
        "tflite_model_file": "quantized_tflite_model_file",
        "quantize_to_int16": data["quantize_to_int16"],
    }


@pytest.mark.ci
def test_spilled_input_copies_are_loaded_again(torq_compiler, tmp_path, torch_ops_mlir):
    """A spilled copy of the input must be loaded again from the input, not stored to XRAM: the
    LRAM plan swaps it in with its load, and such a swap-out stores nothing."""
    model = torch_ops_mlir(GELU_MODEL)
    stderr = compile_stderr(
        torq_compiler, tmp_path, model, "SL2610", ["--torq-vm-print-statistics"]
    )
    reloads = re.findall(r"Swap-ins reloaded from the source: (\d+)", stderr)
    assert reloads, "no swap statistics printed"
    assert sum(int(count) for count in reloads) > 0


@pytest.mark.ci
def test_input_bitcast_is_a_view_of_the_loaded_input(torq_compiler, tmp_path, torch_ops_mlir):
    """The GELU tables read the input as i16: that must be a view of the bf16 input tile in
    LRAM, so that the tile is loaded once, not a load of a bitcast of the input in XRAM."""
    model = torch_ops_mlir(GELU_MODEL)
    ir = dispatch_after(torq_compiler, tmp_path, model, "SR200", "torq-fold-convert", [])
    bitcasts = re.findall(r'"torq_hl\.bitcast"\(%\S+\) : \((.*?)\) ->', ir)
    # Without this, the test would pass even if the GELU stopped using bitcasts.
    assert bitcasts, "no bitcast left, so this test proves nothing"
    xram = [t for t in bitcasts if "mem_space = lram" not in t]
    assert not xram, f"bitcasts of XRAM data: {xram}"


@pytest.mark.ci
def test_tiles_write_their_part_of_the_xram_result(torq_compiler, tmp_path, tflite_mlir_model_file):
    """The tiles of the residual and of the block-2 result must write their part of the XRAM
    tensor from inside their loops. A store after the tiles means a concat was moved to LRAM and
    stored back: whole, or as rows that no tile wrote yet."""
    dispatches = dispatches_after(
        torq_compiler, tmp_path, tflite_mlir_model_file.file_path, "SL2610", "torq-fold-convert", []
    )
    ir = "\n".join(dispatches)
    # Without these, the test would pass even if tile-and-fuse stopped building the tile loop, or
    # if CONVERT_RE stopped matching.
    assert re.search(r"= scf\.for ", ir), "no tile loop, so this test proves nothing"
    converts = CONVERT_RE.findall(ir)
    assert converts, "no torq_hl.convert found, so this test proves nothing"
    # Converts inside the tile loops are indented more than the body of the dispatch function.
    top_level = re.finditer(r"^  %\S+ " + CONVERT_RE.pattern, ir, re.M)
    stores = [
        m.group(1)
        for m in top_level
        if "mem_space = lram" in m.group(2) and "mem_space = lram" not in m.group(1)
    ]
    assert not stores, f"stores after the tiles: {stores}"


@pytest.mark.ci
def test_tile_chain_read_by_the_next_tile_stays_in_xram(torq_compiler, tmp_path):
    """The tail tile reads its init from the result of the tiles before it. So the chain of
    inserts that builds the result must stay in XRAM: an LRAM copy loads the parts again and
    stores the whole result."""
    ir = dispatch_after(
        torq_compiler, tmp_path, MATMUL_TAIL_MODEL, "SL2610", "torq-fold-convert", []
    )
    inserts = re.findall(r"= tensor\.insert_slice .* into (tensor<.*>)$", ir, re.M)
    # Without this, the test would pass even if the matmul stopped building its result in parts.
    assert inserts, "no insert_slice found, so this test proves nothing"
    lram = [t for t in inserts if "mem_space = lram" in t]
    assert not lram, f"inserts into LRAM tensors: {lram}"
