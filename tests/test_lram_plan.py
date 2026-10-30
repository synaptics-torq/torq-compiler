import re
import subprocess
from pathlib import Path

import pytest

# Crafted functions at the memref level, run through torq-assign-lram-addresses alone with
# re-planning, as the real pipeline runs it. Sizes are for SL2610 (523008 usable bytes):
# buffers of 8 KB and more are placed top-down, smaller ones bottom-up.

# %x is deferred at its alloc: the buffers before it leave no room. At its store, %p1 and
# %p2 are pinned by program B, and the holes %q1 and %q2 left are each smaller than %x, so
# the plan fails there. The latest op where %p1, %p2 and B's code were movable is the alloc
# of %x, so the forced compaction lands there. Program A pins %q1 across that op, so the
# repack keeps %q1 between %p1 and %p2 and the plan fails again; the compaction is then
# told to place %x itself, at an op where %x is not born yet.
COMPACTION_AT_DEFERRED_BIRTH = """
#lram = #torq_hl<enc mem_space = lram>
func.func @f() {
  %src = "torq_hl.map_binding"() <{binding_index = 0 : index, is_read_only = true, is_write_only = false, offset = 0 : index}> : () -> memref<143360xi8>
  %out = "torq_hl.map_binding"() <{binding_index = 1 : index, is_read_only = false, is_write_only = true, offset = 0 : index}> : () -> memref<143360xi8>
  %p1 = memref.alloc() : memref<153600xi8, #lram>
  %q1 = memref.alloc() : memref<133120xi8, #lram>
  %p2 = memref.alloc() : memref<112640xi8, #lram>
  %q2 = memref.alloc() : memref<81920xi8, #lram>
  %prog_a = torq_hl.program "pin_q1" : !torq_hl.program<slice> {
  ^bb0(%arg0: memref<133120xi8, #lram>):
    torq_hl.return
  }
  %inv_a, %code_a = torq_hl.create_invocation "pin_q1" program(%prog_a : !torq_hl.program<slice>) on 0 : !torq_hl.invocation<slice>, memref<5120xi8>
  %prog_b = torq_hl.program "pin_p1_p2" : !torq_hl.program<slice> {
  ^bb0(%arg0: memref<153600xi8, #lram>, %arg1: memref<112640xi8, #lram>):
    torq_hl.return
  }
  %inv_b, %code_b = torq_hl.create_invocation "pin_p1_p2" program(%prog_b : !torq_hl.program<slice>) on 1 : !torq_hl.invocation<slice>, memref<5120xi8>
  %ca = memref.alloc() : memref<5120xi8, #lram>
  torq_hl.load %code_a : memref<5120xi8> to %ca : memref<5120xi8, #lram> {element_size_bytes = 5120 : i64, input_strides_bytes = array<i64>, shape = array<i64>}
  %cb = memref.alloc() : memref<5120xi8, #lram>
  torq_hl.load %code_b : memref<5120xi8> to %cb : memref<5120xi8, #lram> {element_size_bytes = 5120 : i64, input_strides_bytes = array<i64>, shape = array<i64>}
  torq_hl.start_program %inv_a : !torq_hl.invocation<slice> args(%q1 : memref<133120xi8, #lram>) code(%ca : memref<5120xi8, #lram>) {arg_accesses = [1 : i32]}
  %x = memref.alloc() : memref<143360xi8, #lram>
  torq_hl.start_program %inv_b : !torq_hl.invocation<slice> args(%p1, %p2 : memref<153600xi8, #lram>, memref<112640xi8, #lram>) code(%cb : memref<5120xi8, #lram>) {arg_accesses = [1 : i32, 1 : i32]}
  torq_hl.wait_program %inv_a : !torq_hl.invocation<slice>
  memref.dealloc %ca : memref<5120xi8, #lram>
  memref.dealloc %q1 : memref<133120xi8, #lram>
  memref.dealloc %q2 : memref<81920xi8, #lram>
  torq_hl.load %src : memref<143360xi8> to %x : memref<143360xi8, #lram> {element_size_bytes = 143360 : i64, input_strides_bytes = array<i64>, shape = array<i64>}
  torq_hl.store %x : memref<143360xi8, #lram> to %out : memref<143360xi8> {element_size_bytes = 143360 : i64, output_strides_bytes = array<i64>, shape = array<i64>}
  torq_hl.wait_program %inv_b : !torq_hl.invocation<slice>
  memref.dealloc %p1 : memref<153600xi8, #lram>
  memref.dealloc %p2 : memref<112640xi8, #lram>
  memref.dealloc %cb : memref<5120xi8, #lram>
  memref.dealloc %x : memref<143360xi8, #lram>
  return
}
"""


# %a is deferred at its alloc: %b and %c fill LRAM. The death of %b frees the space, but a
# death releases it after the deferred fills were tried at its op; the death of %c is the
# next op with an event, where %a fits, before its load.
DEFERRED_FILL_BEFORE_LOAD = """
#lram = #torq_hl<enc mem_space = lram>
func.func @f() {
  %src = "torq_hl.map_binding"() <{binding_index = 0 : index, is_read_only = true, is_write_only = false, offset = 0 : index}> : () -> memref<102400xi8>
  %out_b = "torq_hl.map_binding"() <{binding_index = 1 : index, is_read_only = false, is_write_only = true, offset = 0 : index}> : () -> memref<409600xi8>
  %out_c = "torq_hl.map_binding"() <{binding_index = 2 : index, is_read_only = false, is_write_only = true, offset = 0 : index}> : () -> memref<65536xi8>
  %out_a = "torq_hl.map_binding"() <{binding_index = 3 : index, is_read_only = false, is_write_only = true, offset = 0 : index}> : () -> memref<102400xi8>
  %b = memref.alloc() : memref<409600xi8, #lram>
  %c = memref.alloc() : memref<65536xi8, #lram>
  %a = memref.alloc() : memref<102400xi8, #lram>
  torq_hl.store %b : memref<409600xi8, #lram> to %out_b : memref<409600xi8> {element_size_bytes = 409600 : i64, output_strides_bytes = array<i64>, shape = array<i64>}
  memref.dealloc %b : memref<409600xi8, #lram>
  torq_hl.store %c : memref<65536xi8, #lram> to %out_c : memref<65536xi8> {element_size_bytes = 65536 : i64, output_strides_bytes = array<i64>, shape = array<i64>}
  memref.dealloc %c : memref<65536xi8, #lram>
  torq_hl.load %src : memref<102400xi8> to %a : memref<102400xi8, #lram> {element_size_bytes = 102400 : i64, input_strides_bytes = array<i64>, shape = array<i64>}
  torq_hl.store %a : memref<102400xi8, #lram> to %out_a : memref<102400xi8> {element_size_bytes = 102400 : i64, output_strides_bytes = array<i64>, shape = array<i64>}
  memref.dealloc %a : memref<102400xi8, #lram>
  return
}
"""


# %big is 8 KB or more, so it is placed top-down, from the end of the usable LRAM.
LARGE_BUFFER = """
#lram = #torq_hl<enc mem_space = lram>
func.func @f() {
  %src = "torq_hl.map_binding"() <{binding_index = 0 : index, is_read_only = true, is_write_only = false, offset = 0 : index}> : () -> memref<16384xi8>
  %out = "torq_hl.map_binding"() <{binding_index = 1 : index, is_read_only = false, is_write_only = true, offset = 0 : index}> : () -> memref<16384xi8>
  %big = memref.alloc() : memref<16384xi8, #lram>
  torq_hl.load %src : memref<16384xi8> to %big : memref<16384xi8, #lram> {element_size_bytes = 16384 : i64, input_strides_bytes = array<i64>, shape = array<i64>}
  torq_hl.store %big : memref<16384xi8, #lram> to %out : memref<16384xi8> {element_size_bytes = 16384 : i64, output_strides_bytes = array<i64>, shape = array<i64>}
  memref.dealloc %big : memref<16384xi8, #lram>
  return
}
"""


def assign_lram_addresses(torq_compiler, tmp_path, name, ir, options="replan=true"):
    """Run torq-assign-lram-addresses with `options` on `ir` and return the process."""
    iree_opt = Path(torq_compiler.file_path).parent / "iree-opt"
    model = tmp_path / f"{name}.mlir"
    model.write_text(ir)
    cmd = [
        str(iree_opt),
        str(model),
        "--torq-hw=SL2610",
        f"--pass-pipeline=builtin.module(func.func(torq-assign-lram-addresses{{{options}}}))",
    ]
    print("Running:", " ".join(cmd))
    return subprocess.run(cmd, capture_output=True, text=True)


@pytest.mark.ci
def test_forced_compaction_at_the_alloc_leaves_the_buffer_unborn(torq_compiler, tmp_path):
    """A forced compaction at the alloc of the buffer it is told to place runs before the
    buffer is born, so it must not place it. Here nothing else can, and the plan must fail
    with its diagnostic rather than place the buffer twice."""
    proc = assign_lram_addresses(
        torq_compiler, tmp_path, "compaction-at-deferred-birth", COMPACTION_AT_DEFERRED_BIRTH
    )
    assert proc.returncode != 0
    assert "failed to allocate LRAM addresses" in proc.stderr, proc.stderr


@pytest.mark.ci
def test_deferred_fill_waits_for_its_load(torq_compiler, tmp_path):
    """A deferred fill placed before its load would be swapped in without the load, which
    was cancelled at the alloc. It must stay out until an op after the load: here its
    reload at the store, which runs the load."""
    proc = assign_lram_addresses(
        torq_compiler, tmp_path, "deferred-fill-before-load", DEFERRED_FILL_BEFORE_LOAD
    )
    assert proc.returncode == 0, proc.stderr
    body = proc.stdout
    loads = [line for line in body.splitlines() if "torq_hl.load %0 " in line]
    assert len(loads) == 1, body
    # the fill now runs after both deaths
    second_dealloc = body.index("memref.dealloc", body.index("memref.dealloc") + 1)
    assert body.index(loads[0]) > second_dealloc, body


@pytest.mark.ci
def test_pool_not_a_multiple_of_8_keeps_addresses_aligned(torq_compiler, tmp_path):
    """A tile-fit probe at a reduced budget can get a pool that is not a multiple of 8:
    40% of SL2610's LRAM is 209715 bytes. A buffer placed top-down from the end of that
    pool must still get an 8-byte aligned address."""
    proc = assign_lram_addresses(
        torq_compiler, tmp_path, "pool-not-multiple-of-8", LARGE_BUFFER,
        options="replan=true lram-size=209715",
    )
    assert proc.returncode == 0, proc.stderr
    addresses = [int(a) for a in re.findall(r"lram_address = (\d+)", proc.stdout)]
    assert addresses, proc.stdout
    assert all(a % 8 == 0 for a in addresses), addresses
