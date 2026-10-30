import re
import subprocess
from pathlib import Path

import pytest

# The compile-time profile of an NSS code block load must use the size of the block, which
# torq-assign-nss-programs-addresses measures and cuts the DMA to, not the 640-byte upper bound
# the block is created with.

MODEL = Path(__file__).parent / "testdata/torch_ops/layernorm-nss-1x128x768-bf16.mlir"


@pytest.mark.ci
def test_code_loads_profiled_with_block_size(torq_compiler, tmp_path):
    proc = subprocess.run(
        [
            str(torq_compiler.file_path),
            str(MODEL),
            "-o",
            str(tmp_path / "model.vmfb"),
            "--torq-hw=SL2610",
            "--torq-target-host-triple=native",
            "--torq-enable-profiling",
            "--mlir-print-ir-after=torq-profiling",
            "--mlir-disable-threading",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    ir = proc.stderr

    # the code blocks, %value -> size in bytes
    blocks = dict(re.findall(r"(%\w+) = torq_hl\.get_block .* -> memref<(\d+)xi8>", ir))
    assert blocks

    # the bytes profiled for each DMA that loads one of them
    loads = re.findall(
        r"dma_in_cfg from (%\w+) :.*\n\s*torq_hw\.dma_in_start \{\"torq-async-bytes\" = (\d+)", ir
    )
    code_loads = [(blocks[src], size) for src, size in loads if src in blocks]
    assert code_loads
    assert all(block == size for block, size in code_loads), code_loads
    assert any(int(size) < 640 for _, size in code_loads)
