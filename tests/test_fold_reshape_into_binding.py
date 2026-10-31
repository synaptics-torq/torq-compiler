import re
import subprocess
from pathlib import Path

import pytest

from .torch_ops_models import torch_ops_mlir

# IR checks for the fold of a reshape that sits between a dispatch result and its
# dispatch.tensor.store. Without the fold, bufferization writes the result to a temporary
# buffer and leaves a torq_hl.host_copy of the reshaped view to the binding, which becomes a
# full load and store of the output. The numerics are the same either way, so only the IR
# shows it.

TESTDATA = Path(__file__).parent / "testdata"

CASES = [
    # unit-dim collapse_shape after a depthwise conv
    ("gen", "depthwise-conv1d-bf16-c464-k5.mlir"),
    # collapse_shape after depth-to-space
    TESTDATA / "tosa_ops" / "d2s_8x8x128xi8_crd.mlir",
]


def host_copies_from_reshapes(torq_compiler, tmp_path, model):
    """Compile model and return the torq_hl.host_copy ops of the dispatch, right before
    LowerHostCopiesToNpu, whose source is a memref.collapse_shape or memref.expand_shape."""
    cmd = [
        str(torq_compiler.file_path),
        str(model),
        "-o",
        str(tmp_path / "model.vmfb"),
        "--torq-hw=SL2610",
        "--torq-target-host-triple=native",
        "--mlir-disable-threading",
        "--mlir-print-ir-before=torq-lower-host-copies-to-npu",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"compile failed:\n{proc.stderr[-4000:]}"

    # The tile-fit probe compiles extracted copies of the ops; only the dispatch itself matters.
    dumps = [
        d
        for d in proc.stderr.split("// -----// IR Dump Before ")
        if "_dispatch_" in d and "extracted_ops" not in d
    ]
    assert dumps, "no dump of the dispatch before torq-lower-host-copies-to-npu"

    found = []
    for dump in dumps:
        reshaped = set(re.findall(r"(%[\w.]+) = memref\.(?:collapse|expand)_shape", dump))
        for line in dump.splitlines():
            copy = re.search(r"torq_hl\.host_copy (%[\w.]+)", line)
            if copy and copy.group(1) in reshaped:
                found.append(line.strip())
    return found


@pytest.mark.ci
@pytest.mark.parametrize("model", CASES, ids=lambda p: p[1] if isinstance(p, tuple) else p.name)
def test_output_reshape_is_not_copied(torq_compiler, tmp_path, torch_ops_mlir, model):
    """The reshape folds into the binding, so the result is written in place and no copy of a
    reshaped view is left for LowerHostCopiesToNpu."""
    if isinstance(model, tuple):
        _, name = model
        model = torch_ops_mlir(name)
    found = host_copies_from_reshapes(torq_compiler, tmp_path, model)
    assert not found, "output copied through a reshape:\n" + "\n".join(found)
