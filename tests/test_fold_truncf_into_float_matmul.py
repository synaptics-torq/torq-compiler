import re
import subprocess
from pathlib import Path

import pytest

# IR check for a bf16 matmul that torch-mlir accumulates in f32 and then truncates to bf16.
# The truncation must fold into the matmul, so the matmul writes bf16 directly. Otherwise the
# full f32 result goes out between the matmul and a separate conversion, which doubles the DMA
# traffic and shrinks the tiles. The numerics are the same either way, so only the IR shows it.

TESTDATA = Path(__file__).parent / "testdata"

CASES = [
    # rank-4 input times a rank-2 weight, lowered to linalg.batch_matmul
    TESTDATA / "torch_ops" / "matmul-rank4x2-bf16-f32acc.mlir",
    # two rank-2 operands, lowered to linalg.matmul
    TESTDATA / "torch_ops" / "matmul-rank2-bf16-f32acc.mlir",
]


@pytest.mark.ci
@pytest.mark.parametrize("model", CASES, ids=lambda p: p.name)
# runtime_hw_type is unused, but requesting it makes the plugin run the test for each chip.
def test_matmul_writes_bf16(torq_compiler, runtime_hw_type, chip_config, tmp_path, model):
    cmd = [
        str(torq_compiler.file_path),
        str(model),
        "-o",
        str(tmp_path / "model.vmfb"),
        f"--torq-hw={chip_config.data['target']}",
        "--torq-target-host-triple=native",
        "--mlir-disable-threading",
        "--mlir-print-ir-after=torq-optimize-linalg-for-torq",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"compile failed:\n{proc.stderr[-4000:]}"

    # The tile-fit probe compiles extracted copies of the ops; only the dispatch itself matters.
    dumps = [
        d
        for d in proc.stderr.split("// -----// IR Dump After ")
        if "_dispatch_" in d and "extracted_ops" not in d
    ]
    assert dumps, "no dump of the dispatch after torq-optimize-linalg-for-torq"

    matmuls = [
        line.strip()
        for dump in dumps
        for line in dump.splitlines()
        if re.search(r"linalg\.(batch_)?matmul ", line)
    ]
    assert matmuls, "no matmul in the dispatch"
    f32 = [line for line in matmuls if re.search(r"-> tensor<[0-9x]+xf32>", line)]
    assert not f32, "matmul still writes f32:\n" + "\n".join(f32)
    assert not any("arith.truncf" in dump for dump in dumps), "separate truncf left"
