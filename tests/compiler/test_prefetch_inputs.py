import subprocess
from pathlib import Path

import pytest

# IR checks for torq-prefetch-inputs, the stream scheduler.

# Two slice programs. The second reads the first's buffer through a torq_hl.reinterpret_cast
# (the bufferized torq_hl.bitcast) and needs a 100000 B constant, which can load while the
# first program runs. The view sits between the first wait_program and the constant load, where
# a barrier would keep the load behind the wait.
VIEW_BETWEEN_PROGRAMS = """
#lram = #torq_hl<enc mem_space = lram>
func.func @f() {
  %w = "torq_hl.const"() <{value = dense<1> : tensor<100000xi8>}> : () -> memref<100000xi8>
  %big = memref.alloc() : memref<400000xi8, #lram>
  %p0 = torq_hl.program "p0" : !torq_hl.program<slice> {
  ^bb0(%arg0: memref<400000xi8, #lram>):
    torq_hw.slice_profiling
      memory : 0 bytes
      memory_info : ""
      max_cycle : 100
      ndl_cycles : dedr(0) dewr(0) debr(0) deqw(0) cedw(0) cedr(0) ceww(0) cewr(0) cepr(0) acbw(0) acbr(0) acpr(0)
    torq_hl.return
  }
  %inv0, %code0 = torq_hl.create_invocation "p0" program(%p0 : !torq_hl.program<slice>) on 0 : !torq_hl.invocation<slice>, memref<16xi8>
  %c0 = memref.alloc() : memref<16xi8, #lram>
  torq_hl.load %code0 : memref<16xi8> to %c0 : memref<16xi8, #lram> {element_size_bytes = 16 : i64, input_strides_bytes = array<i64>, shape = array<i64>, "torq-async-duration-cycles" = 1 : i64}
  torq_hl.start_program %inv0 : !torq_hl.invocation<slice> args(%big : memref<400000xi8, #lram>) code(%c0 : memref<16xi8, #lram>) {arg_accesses = [2 : i32]}
  torq_hl.wait_program %inv0 : !torq_hl.invocation<slice>
  %view = torq_hl.reinterpret_cast %big to offset: [0], sizes: [400000], strides: [1] : memref<400000xi8, #lram> to memref<400000xi8, #lram>
  %a = memref.alloc() : memref<100000xi8, #lram>
  torq_hl.load %w : memref<100000xi8> to %a : memref<100000xi8, #lram> {element_size_bytes = 100000 : i64, input_strides_bytes = array<i64>, shape = array<i64>, "torq-async-duration-cycles" = 30 : i64}
  %p1 = torq_hl.program "p1" : !torq_hl.program<slice> {
  ^bb0(%arg0: memref<400000xi8, #lram>, %arg1: memref<100000xi8, #lram>):
    torq_hw.slice_profiling
      memory : 0 bytes
      memory_info : ""
      max_cycle : 100
      ndl_cycles : dedr(0) dewr(0) debr(0) deqw(0) cedw(0) cedr(0) ceww(0) cewr(0) cepr(0) acbw(0) acbr(0) acpr(0)
    torq_hl.return
  }
  %inv1, %code1 = torq_hl.create_invocation "p1" program(%p1 : !torq_hl.program<slice>) on 0 : !torq_hl.invocation<slice>, memref<16xi8>
  %c1 = memref.alloc() : memref<16xi8, #lram>
  torq_hl.load %code1 : memref<16xi8> to %c1 : memref<16xi8, #lram> {element_size_bytes = 16 : i64, input_strides_bytes = array<i64>, shape = array<i64>, "torq-async-duration-cycles" = 1 : i64}
  torq_hl.start_program %inv1 : !torq_hl.invocation<slice> args(%view, %a : memref<400000xi8, #lram>, memref<100000xi8, #lram>) code(%c1 : memref<16xi8, #lram>) {arg_accesses = [3 : i32, 1 : i32]}
  torq_hl.wait_program %inv1 : !torq_hl.invocation<slice>
  return
}
"""


@pytest.mark.ci
def test_constant_load_moves_over_a_view(torq_compiler, tmp_path):
    """The constant load moves in front of the first wait_program: the view between them is
    not a barrier."""
    iree_opt = Path(torq_compiler.file_path).parent / "iree-opt"
    model = tmp_path / "model.mlir"
    model.write_text(VIEW_BETWEEN_PROGRAMS)
    proc = subprocess.run(
        [
            str(iree_opt),
            str(model),
            "--torq-hw=SL2610",
            "--pass-pipeline=builtin.module(func.func(torq-prefetch-inputs))",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    load = next(i for i, line in enumerate(lines) if "memref<100000xi8> to" in line)
    wait = next(i for i, line in enumerate(lines) if "torq_hl.wait_program" in line)
    assert load < wait, "the constant load did not move in front of the first wait_program"
