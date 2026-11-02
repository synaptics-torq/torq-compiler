import subprocess
from pathlib import Path

import pytest

# IR checks for torq-pair-dma-channels: an input DMA is pulled up into the task of an
# earlier output DMA so the two channels run at the same time. It must not be pulled
# over the wait of a slice program that is still running and reads the LRAM range the
# input writes. The simulator runs everything in order, so only the board shows that
# race; this checks the decision in the IR instead.

# A slice program starts, an output runs while it computes, then the slice is waited
# for and an input follows. The slice reads LRAM 1024..2048.
SLICE_RUNNING_ACROSS_OUTPUT = """
#lram = #torq_hl<enc mem_space = lram>
func.func @f(%inv: !torq_hl.invocation<slice>, %x: memref<1024xi8>, %y: memref<1024xi8>) {
  %prog = memref.alloc() {lram_address = 0 : i64} : memref<64xi8, #lram>
  %sbuf = memref.alloc() {lram_address = 1024 : i64} : memref<1024xi8, #lram>
  %obuf = memref.alloc() {lram_address = 4096 : i64} : memref<1024xi8, #lram>
  %ibuf = memref.alloc() {lram_address = IN_ADDR : i64} : memref<1024xi8, #lram>
  torq_hw.nss_task {"torq-task-id" = 0 : i64}() {
    "torq_hw.slice_start"(%inv, %prog, %sbuf) <{id = 0 : index, program_address = 0 : i32}> : (!torq_hl.invocation<slice>, memref<64xi8, #lram>, memref<1024xi8, #lram>) -> ()
  }
  torq_hw.nss_task {"torq-task-id" = 1 : i64}() {
    torq_hw.dma_out_cfg from %obuf : memref<1024xi8, #lram> to %y : memref<1024xi8> {read_address = 4096 : i32, read_ndl = #torq_hw<dma_ndl{dims = [[1024, 1]]}>, write_address = 65536 : i32, write_ndl = #torq_hw<dma_ndl{dims = [[1024, 1]]}>}
    torq_hw.dma_out_start
    torq_hw.dma_out_wait
  }
  torq_hw.nss_task {"torq-task-id" = 2 : i64}() {
    "torq_hw.slice_wait"() <{id = 0 : index}> : () -> ()
  }
  torq_hw.nss_task {"torq-task-id" = 3 : i64}() {
    torq_hw.dma_in_cfg from %x : memref<1024xi8> to %ibuf : memref<1024xi8, #lram> {read_address = 131072 : i32, read_ndl = #torq_hw<dma_ndl{dims = [[1024, 1]]}>, write_address = IN_ADDR : i32, write_ndl = #torq_hw<dma_ndl{dims = [[1024, 1]]}>}
    torq_hw.dma_in_start
    torq_hw.dma_in_wait
  }
  return
}
"""


def run_pass(torq_compiler, tmp_path, in_addr):
    """Run torq-pair-dma-channels on SLICE_RUNNING_ACROSS_OUTPUT with the input at
    in_addr and return the number of NSS tasks left."""
    iree_opt = Path(torq_compiler.file_path).parent / "iree-opt"
    model = tmp_path / f"in-{in_addr}.mlir"
    model.write_text(SLICE_RUNNING_ACROSS_OUTPUT.replace("IN_ADDR", str(in_addr)))
    proc = subprocess.run(
        [
            str(iree_opt),
            str(model),
            "--pass-pipeline=builtin.module(func.func(torq-pair-dma-channels))",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return proc.stdout.count("torq_hw.nss_task")


@pytest.mark.ci
def test_input_not_moved_over_wait_of_slice_reading_it(torq_compiler, tmp_path):
    """The input writes the LRAM range the running slice reads, so it stays after the
    slice_wait and the four tasks stay apart."""
    assert run_pass(torq_compiler, tmp_path, 1024) == 4


@pytest.mark.ci
def test_input_moved_over_wait_of_unrelated_slice(torq_compiler, tmp_path):
    """The input writes LRAM the slice does not touch, so it joins the output task."""
    assert run_pass(torq_compiler, tmp_path, 8192) == 3
