import re
import subprocess
from pathlib import Path

import pytest

# IR checks for torq-prefetch-constant-loads: a constant load moves in front of the
# wait_program of the program before it, so its DMA runs while that program
# computes.

COMMON_FLAGS = [
    "--torq-hw=SL2610",
    "--torq-target-host-triple=native",
    "--torq-disable-host",
    "--torq-disable-css",
    "--mlir-disable-threading",
    "--mlir-print-ir-after=torq-prefetch-constant-loads",
]

# Two 1x1 convs. The weights of the second conv are a constant that can load while
# the first conv runs.
TWO_CONVS = """
module {
  func.func @main(%x: tensor<1x28x28x64xi8>) -> tensor<1x28x28x64xi8> {
    %w1 = "tosa.const"() <{values = dense<3> : tensor<64x1x1x64xi8>}> : () -> tensor<64x1x1x64xi8>
    %w2 = "tosa.const"() <{values = dense<5> : tensor<64x1x1x64xi8>}> : () -> tensor<64x1x1x64xi8>
    %b = "tosa.const"() <{values = dense<7> : tensor<64xi32>}> : () -> tensor<64xi32>
    %zp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %m = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %s = "tosa.const"() <{values = dense<38> : tensor<1xi8>}> : () -> tensor<1xi8>
    %z32 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %c1 = tosa.conv2d %x, %w1, %b, %zp, %zp {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x28x28x64xi8>, tensor<64x1x1x64xi8>, tensor<64xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x28x28x64xi32>
    %r1 = tosa.rescale %c1, %m, %s, %z32, %zp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x28x28x64xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x28x28x64xi8>
    %c2 = tosa.conv2d %r1, %w2, %b, %zp, %zp {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x28x28x64xi8>, tensor<64x1x1x64xi8>, tensor<64xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x28x28x64xi32>
    %r2 = tosa.rescale %c2, %m, %s, %z32, %zp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x28x28x64xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x28x28x64xi8>
    return %r2 : tensor<1x28x28x64xi8>
  }
}
"""


def constant_loads_while_running(torq_compiler, tmp_path, extra_flags):
    """Compile TWO_CONVS and return how many loads of a torq_hl.const the NSS issues
    while a program runs (between its start_program and its wait_program)."""
    model = tmp_path / "model.mlir"
    model.write_text(TWO_CONVS)
    cmd = [str(torq_compiler.file_path), str(model), "-o", str(tmp_path / "model.vmfb")]
    cmd += COMMON_FLAGS + extra_flags
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"compile failed:\n{proc.stderr}"
    dumps = [d for d in proc.stderr.split("// -----// IR Dump After") if "func.func @main_dispatch" in d]
    assert dumps, "no dump of the dispatch after torq-prefetch-constant-loads"
    dump = dumps[-1]

    consts = set(re.findall(r'(%\w+) = "torq_hl\.const"', dump))
    views = {}
    running = set()
    count = 0
    for line in dump.splitlines():
        view = re.match(r"\s*(%\w+) = memref\.(?:subview|collapse_shape|expand_shape) (%\w+)", line)
        if view:
            views[view.group(1)] = view.group(2)
        start = re.search(r"torq_hl\.start_program (%\w+)", line)
        if start:
            running.add(start.group(1))
        wait = re.search(r"torq_hl\.wait_program (%\w+)", line)
        if wait:
            running.discard(wait.group(1))
        load = re.search(r"torq_hl\.load (%\w+)", line)
        if load and running:
            source = load.group(1)
            while source in views:
                source = views[source]
            count += source in consts
    return count


@pytest.mark.ci
def test_constant_load_overlaps_previous_program(torq_compiler, tmp_path):
    """The weights of the second conv load while the first conv runs."""
    off = constant_loads_while_running(
        torq_compiler, tmp_path, ["--torq-constant-prefetch=false"]
    )
    on = constant_loads_while_running(torq_compiler, tmp_path, [])
    assert on > off, f"no constant load moved in front of a wait_program (on={on}, off={off})"


# NSS-level IR: a 400000 B buffer stays live while a 100000 B constant loads for
# the second program. The LRAM holds the buffer and one copy of the constant, but
# not two copies.
LIVE_BUFFER_AND_WEIGHT = """
#lram = #torq_hl<enc mem_space = lram>
func.func @f() {
  %w = "torq_hl.const"() <{value = dense<1> : tensor<100000xi8>}> : () -> memref<100000xi8>
  %big = memref.alloc() : memref<400000xi8, #lram>
  %p0 = torq_hl.program "p0" : !torq_hl.program<slice> {
  ^bb0(%arg0: memref<400000xi8, #lram>):
    torq_hl.return
  }
  %inv0, %code0 = torq_hl.create_invocation "p0" program(%p0 : !torq_hl.program<slice>) on 0 : !torq_hl.invocation<slice>, memref<16xi8>
  %c0 = memref.alloc() : memref<16xi8, #lram>
  torq_hl.load %code0 : memref<16xi8> to %c0 : memref<16xi8, #lram> {element_size_bytes = 16 : i64, input_strides_bytes = array<i64>, shape = array<i64>}
  torq_hl.start_program %inv0 : !torq_hl.invocation<slice> args(%big : memref<400000xi8, #lram>) code(%c0 : memref<16xi8, #lram>) {arg_accesses = [2 : i32]}
  torq_hl.wait_program %inv0 : !torq_hl.invocation<slice>
  memref.dealloc %c0 : memref<16xi8, #lram>
  %a = memref.alloc() : memref<100000xi8, #lram>
  torq_hl.load %w : memref<100000xi8> to %a : memref<100000xi8, #lram> {element_size_bytes = 100000 : i64, input_strides_bytes = array<i64>, shape = array<i64>}
  %p1 = torq_hl.program "p1" : !torq_hl.program<slice> {
  ^bb0(%arg0: memref<400000xi8, #lram>, %arg1: memref<100000xi8, #lram>):
    torq_hl.return
  }
  %inv1, %code1 = torq_hl.create_invocation "p1" program(%p1 : !torq_hl.program<slice>) on 0 : !torq_hl.invocation<slice>, memref<16xi8>
  %c1 = memref.alloc() : memref<16xi8, #lram>
  torq_hl.load %code1 : memref<16xi8> to %c1 : memref<16xi8, #lram> {element_size_bytes = 16 : i64, input_strides_bytes = array<i64>, shape = array<i64>}
  torq_hl.start_program %inv1 : !torq_hl.invocation<slice> args(%big, %a : memref<400000xi8, #lram>, memref<100000xi8, #lram>) code(%c1 : memref<16xi8, #lram>) {arg_accesses = [3 : i32, 1 : i32]}
  torq_hl.wait_program %inv1 : !torq_hl.invocation<slice>
  memref.dealloc %c1 : memref<16xi8, #lram>
  memref.dealloc %a : memref<100000xi8, #lram>
  memref.dealloc %big : memref<400000xi8, #lram>
  return
}
"""


@pytest.mark.ci
def test_constant_load_that_fits_once_moves(torq_compiler, tmp_path):
    """The constant moves in front of the first wait_program. The LRAM check counts
    the constant only from its new position to its old alloc, not twice."""
    iree_opt = Path(torq_compiler.file_path).parent / "iree-opt"
    model = tmp_path / "model.mlir"
    model.write_text(LIVE_BUFFER_AND_WEIGHT)
    proc = subprocess.run(
        [
            str(iree_opt),
            str(model),
            "--torq-hw=SL2610",
            "--pass-pipeline=builtin.module(func.func(torq-prefetch-constant-loads))",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    load = next(i for i, line in enumerate(lines) if "memref<100000xi8> to" in line)
    wait = next(i for i, line in enumerate(lines) if "torq_hl.wait_program" in line)
    assert load < wait, "the constant load did not move in front of the first wait_program"
