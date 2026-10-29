import re
import subprocess

import pytest

from torq.testing.iree import MODELS_DIR

# The tanh GELU of gelu-gemma.mlir. On SL2610 it runs untiled on both slices, and its LRAM plan
# spills copies of the input along with computed buffers.
GELU_MODEL = MODELS_DIR / "torch_ops" / "gelu-gemma.mlir"


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


def dispatch_after(torq_compiler, tmp_path, model, hw, pass_arg, extra_flags):
    """Compile `model` and return the IR of its dispatch after `pass_arg`."""
    stderr = compile_stderr(
        torq_compiler, tmp_path, model, hw, [f"--mlir-print-ir-after={pass_arg}"] + extra_flags
    )

    # The tile-fit probe prints its own runs too; keep the dispatch.
    dumps = [
        dump
        for dump in stderr.split("// -----// IR Dump After")
        if re.search(r"func\.func @main_dispatch", dump)
    ]
    assert dumps, f"no dump of the dispatch after {pass_arg}"
    return dumps[-1]


@pytest.mark.ci
def test_spilled_input_copies_are_loaded_again(torq_compiler, tmp_path):
    """A spilled copy of the input must be loaded again from the input, not stored to XRAM: the
    LRAM plan swaps it in with its load, and such a swap-out stores nothing."""
    stderr = compile_stderr(
        torq_compiler, tmp_path, GELU_MODEL, "SL2610", ["--torq-vm-print-statistics"]
    )
    reloads = re.findall(r"Swap-ins reloaded from the source: (\d+)", stderr)
    assert reloads, "no swap statistics printed"
    assert sum(int(count) for count in reloads) > 0


@pytest.mark.ci
def test_input_bitcast_is_a_view_of_the_loaded_input(torq_compiler, tmp_path):
    """The GELU tables read the input as i16: that must be a view of the bf16 input tile in
    LRAM, so that the tile is loaded once, not a load of a bitcast of the input in XRAM."""
    ir = dispatch_after(torq_compiler, tmp_path, GELU_MODEL, "SR200", "torq-fold-convert", [])
    bitcasts = re.findall(r'"torq_hl\.bitcast"\(%\S+\) : \((.*?)\) ->', ir)
    # Without this, the test would pass even if the GELU stopped using bitcasts.
    assert bitcasts, "no bitcast left, so this test proves nothing"
    xram = [t for t in bitcasts if "mem_space = lram" not in t]
    assert not xram, f"bitcasts of XRAM data: {xram}"
