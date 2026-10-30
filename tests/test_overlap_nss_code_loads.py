import re
import subprocess
from pathlib import Path

import pytest

# IR checks for torq-overlap-nss-code-loads: an NSS program block that ends by waiting for a
# program loads the next code (the next block of its program, or the first block of the next
# program) before that wait, so the load runs while the program still runs. A block that ends
# with a program running already overlaps its loads and stays as it is.

HEADER = """
#lram = #torq_hl<enc mem_space = lram>
func.func @f() {
  %p = torq_hl.program "nss_program_0" {block_sizes = array<i32: 640, 640>} : !torq_hl.program<nss> {
  ^bb0(%arg0: memref<640xi8, #lram>, %arg1: memref<640xi8, #lram>, %arg2: !torq_hl.invocation<nss>, %arg3: !torq_hl.invocation<nss>, %arg4: !torq_hl.invocation<slice>, %arg5: memref<1024xi8, #lram>, %arg6: memref<1024xi8>):
"""

FOOTER = """
  }
  return
}
"""

START = """    torq_hl.start_program %arg4 : !torq_hl.invocation<slice> args(%arg5 : memref<1024xi8, #lram>) code(%arg5 : memref<1024xi8, #lram>)
"""
WAIT = """    torq_hl.wait_program %arg4 : !torq_hl.invocation<slice>
"""
STORE = """    torq_hl.store %arg5 : memref<1024xi8, #lram> to %arg6 : memref<1024xi8> {element_size_bytes = 1024 : i64, output_strides_bytes = array<i64>, shape = array<i64>}
"""


def load_code(name, invocation, block, area):
    return f"""    %{name} = torq_hl.get_block {invocation} block {block} : !torq_hl.invocation<nss> -> memref<640xi8>
    torq_hl.load %{name} : memref<640xi8> to {area} : memref<640xi8, #lram> {{element_size_bytes = 640 : i64, input_strides_bytes = array<i64>, shape = array<i64>}}
"""


NEXT = """    torq_hl.next lram_area(%arg1 : memref<640xi8, #lram>) to ^bb1
  ^bb1:
"""
RETURN = """    torq_hl.return %arg0, %arg1 : memref<640xi8, #lram>, memref<640xi8, #lram>
"""


def run_pass(torq_compiler, tmp_path, body):
    """Run torq-overlap-nss-code-loads on a program with this body and return its op names, one
    list per block."""
    iree_opt = Path(torq_compiler.file_path).parent / "iree-opt"
    model = tmp_path / "in.mlir"
    model.write_text(HEADER + body + FOOTER)
    proc = subprocess.run(
        [
            str(iree_opt),
            str(model),
            "--pass-pipeline=builtin.module(func.func(torq-overlap-nss-code-loads))",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    program = proc.stdout.split("torq_hl.program", 1)[1]
    blocks = []
    for line in program.splitlines()[1:]:
        if re.match(r"\s*\^bb", line):
            blocks.append([])
        elif m := re.search(r"torq_hl\.(\w+)", line):
            blocks[-1].append(m.group(1))
            if m.group(1) == "return":
                break
    return blocks


@pytest.mark.ci
def test_code_loads_moved_before_last_wait(torq_compiler, tmp_path):
    """Both blocks end by waiting for the slice: the load of the next block and the load of the
    next program go before the wait, and the store stays after it."""
    body = (
        START + WAIT + load_code("b1", "%arg2", 1, "%arg1") + NEXT
        + START + WAIT + STORE + load_code("p1", "%arg3", 0, "%arg0") + RETURN
    )
    assert run_pass(torq_compiler, tmp_path, body) == [
        ["start_program", "get_block", "load", "wait_program", "next"],
        ["start_program", "get_block", "load", "wait_program", "store", "return"],
    ]


@pytest.mark.ci
def test_code_loads_kept_when_program_running(torq_compiler, tmp_path):
    """The first block ends with the slice running and the second has no start or wait: their
    loads already run as early as they can and stay where they are."""
    body = (
        WAIT + START + load_code("b1", "%arg2", 1, "%arg1") + NEXT
        + STORE + load_code("p1", "%arg3", 0, "%arg0") + RETURN
    )
    assert run_pass(torq_compiler, tmp_path, body) == [
        ["wait_program", "start_program", "get_block", "load", "next"],
        ["store", "get_block", "load", "return"],
    ]
