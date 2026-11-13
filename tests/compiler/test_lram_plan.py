import re
import subprocess
from collections import Counter
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


def assign_lram_addresses(torq_compiler, tmp_path, name, ir, options="replan=true", hw="SL2610"):
    """Run torq-assign-lram-addresses with `options` on `ir` and return the process."""
    iree_opt = Path(torq_compiler.file_path).parent / "iree-opt"
    model = tmp_path / f"{name}.mlir"
    model.write_text(ir)
    cmd = [
        str(iree_opt),
        str(model),
        f"--torq-hw={hw}",
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


def fragmented_sr200_ir(request=51744, background=False, bias=1408):
    """Recreate the NASNet failure geometry, without any concat or model dependency.

    Bias is at 0, code at 14640, output at 66384, weights at 75008 (usable-relative).
    Two 4056-byte pads and scratch code initially occupy [1408, 14640). An async
    scratch program pins them until the target buffers are pinned by `hold`.
    `hold`'s wait and the next code load touch its code's resident range: the old
    lastMovablePosition heuristic never moves that code after the pads are freed.
    """
    lines = ["#lram = #torq_hl<enc mem_space = lram>", "func.func @f() {"]

    def binding(name, index, size, output=False):
        lines.append(
            f'  %{name} = "torq_hl.map_binding"() <{{binding_index = {index} : index, '
            f'is_read_only = {str(not output).lower()}, is_write_only = {str(output).lower()}, '
            f'offset = 0 : index}}> : () -> memref<{size}xi8>'
        )

    pad = (14640 - bias - 5120) // 2
    sizes = {"bias": bias, "pad1": pad, "pad2": pad, "sc": 5120,
        "weights": 185856, "output": 8624, "code": 5120, "x": request}

    def typ(name, lram=True):
        return f"memref<{sizes[name]}xi8{', #lram' if lram else ''}>"

    def load(src, dst):
        lines.append(
            f"  torq_hl.load %{src} : {typ(dst, False)} to %{dst} : {typ(dst)} "
            f"{{element_size_bytes = {sizes[dst]} : i64, input_strides_bytes = array<i64>, "
            "shape = array<i64>}"
        )

    def program(name, args, core):
        types = ", ".join(f"%a{i}: {typ(a)}" for i, a in enumerate(args))
        lines.extend([
            f'  %p_{name} = torq_hl.program "{name}" : !torq_hl.program<slice> {{',
            f"  ^bb0({types}):" if args else "  ^bb0:",
            "    torq_hl.return", "  }",
            f'  %i_{name}, %c_{name} = torq_hl.create_invocation "{name}" '
            f"program(%p_{name} : !torq_hl.program<slice>) on {core} : "
            "!torq_hl.invocation<slice>, memref<5120xi8>",
        ])

    def start(name, args, accesses, code):
        arg_str = ", ".join(f"%{a}" for a in args)
        types = ", ".join(typ(a) for a in args)
        access_str = ", ".join(f"{a} : i32" for a in accesses)
        lines.append(
            f"  torq_hl.start_program %i_{name} : !torq_hl.invocation<slice> "
            f"args({arg_str} : {types}) code(%{code} : {typ(code)}) "
            f"{{arg_accesses = [{access_str}]}}"
        )

    def wait(name):
        lines.append(f"  torq_hl.wait_program %i_{name} : !torq_hl.invocation<slice>")

    def free(name):
        lines.append(f"  memref.dealloc %{name} : {typ(name)}")

    for i, name in enumerate(("bias", "weights", "x")):
        binding(f"src_{name}", i, sizes[name])
    binding("out", 3, 8624, output=True)
    program("scratch", ["pad1", "pad2"], 1)
    # Read/write weights make this a computed live value, not a repeatable fill.
    program("hold", ["bias", "weights", "output"], 0)
    program("consume", ["bias", "weights", "output", "x"], 0)
    if background:
        program("background", ["bias"], 1)
    for name in ("bias", "pad1", "pad2", "sc", "weights", "output", "code"):
        lines.append(f"  %{name} = memref.alloc() : {typ(name)}")
        if name in ("bias", "weights"):
            load(f"src_{name}", name)
    load("c_scratch", "sc")
    load("c_hold", "code")
    start("scratch", ["pad1", "pad2"], [2, 2], "sc")
    lines.append(f"  %x = memref.alloc() : {typ('x')}")
    load("src_x", "x")
    start("hold", ["bias", "weights", "output"], [1, 3, 2], "code")
    wait("scratch")
    if background:
        load("c_background", "sc")
        # Its code adds an obstacle; the bias is also needed by the consumer.
        start("background", ["bias"], [1], "sc")
    free("pad1")
    free("pad2")
    if not background:
        free("sc")
    wait("hold")
    load("c_consume", "code")
    start("consume", ["bias", "weights", "output", "x"], [1, 1, 2, 1], "code")
    wait("consume")
    if background:
        wait("background")
        free("sc")
    lines.append(
        f"  torq_hl.store %output : {typ('output')} to %out : {typ('output', False)} "
        "{element_size_bytes = 8624 : i64, output_strides_bytes = array<i64>, shape = array<i64>}"
    )
    for name in ("bias", "weights", "output", "code", "x"):
        free(name)
    return "\n".join(lines + ["  return", "}"])


def check_repacked_values_and_lifetimes(ir, bias_size=1408):
    """Execute contiguous transfers in a byte-addressed LRAM, checking physical SSA
    lifetimes, overlap, and async pins. Stub programs model writes at their waits;
    this is an allocator/transfer test, not a numerical slice-kernel test.
    """
    aliases = dict(re.findall(r"(#\w+) = #torq_hl<enc mem_space = (\w+)", ir))
    occurrences = Counter(re.findall(r"%\w+", ir))
    memory = bytearray(262144)
    buffers, live, programs, running = {}, set(), {}, {}
    stores, reloads = [], []
    code_sources, code_load_addresses, code_start_addresses = {}, {}, {}
    result = None
    depth = 0

    def program_code(name):
        # Different programs, including equal-length names, need distinct code.
        pattern = name.encode("utf-8") + b"\0"
        return (pattern * ((5120 + len(pattern) - 1) // len(pattern)))[:5120]

    def pinned():
        return {v for args in running.values() for v in args}

    def read(value):
        assert value in live, f"use after dealloc: {value}"
        size, address, data = buffers[value]
        return bytes(memory[address:address + size]) if address is not None else data

    def write(value, data):
        assert value in live and value not in pinned(), f"write to unavailable buffer: {value}"
        size, address, _ = buffers[value]
        assert len(data) == size
        if address is None:
            buffers[value] = size, address, bytes(data)
        else:
            memory[address:address + size] = data

    for line in ir.splitlines():
        if " = torq_hl.program " in line:
            depth = 1
            continue
        if depth:
            depth += line.count("{") - line.count("}")
            continue
        values = re.findall(r"%[\w]+", line)
        if "torq_hl.map_binding" in line:
            size = int(re.search(r"memref<(\d+)xi8", line)[1])
            index = int(re.search(r"binding_index = (\d+)", line)[1])
            data = bytes((i * 13 + index * 29) % 256 for i in range(size))
            buffers[values[0]] = size, None, data
            live.add(values[0])
        elif "torq_hl.create_invocation" in line:
            name = re.search(r'create_invocation "([^"]+)"', line)[1]
            programs[values[0]] = name
            code_sources[values[1]] = name
            buffers[values[1]] = 5120, None, program_code(name)
            live.add(values[1])
        elif "memref.alloc()" in line:
            value = values[0]
            # A deferred fill's original alloc has no users and never becomes physical.
            # It is inert (canonicalization can erase it), not a resident allocation.
            if occurrences[value] == 1:
                continue
            size = int(re.search(r"memref<(\d+)xi8", line)[1])
            address_attr = re.search(r"lram_address = (\d+)", line)
            space = re.search(r"memref<\d+xi8, (#\w+)", line)
            is_lram = "mem_space = lram" in line or (space and aliases.get(space[1]) == "lram")
            assert not is_lram or address_attr, line
            address = int(address_attr[1]) if is_lram else None
            if address is not None:
                assert address % 8 == 0 and 1280 <= address <= 262144 - size
                for other in live:
                    n, a, _ = buffers[other]
                    assert a is None or address + size <= a or a + n <= address, line
            buffers[value] = size, address, bytes(size)
            live.add(value)
        elif "memref.dealloc " in line:
            value = values[0]
            assert value in live and value not in pinned(), line
            live.remove(value)
        elif "torq_hl.load " in line or "torq_hl.store " in line:
            src, dst = values
            data = read(src)
            write(dst, data)
            if src in code_sources:
                code_load_addresses.setdefault(code_sources[src], []).append(buffers[dst][1])
            if "torq_hl.store " in line and buffers[src][1] is not None:
                stores.append(data)
            if "torq_hl.load " in line and buffers[src][1] is None:
                reloads.append(data)
            if "torq_hl.store " in line and buffers[dst][0] == 8624:
                result = data
        elif "torq_hl.start_program " in line:
            invocation, *args = values
            assert invocation not in running
            for value in args:
                read(value)
            name = programs[invocation]
            assert read(args[-1]) == program_code(name), f"program code mismatch: {name}"
            code_start_addresses[name] = buffers[args[-1]][1]
            running[invocation] = args
        elif "torq_hl.wait_program " in line:
            invocation = values[0]
            args = running.pop(invocation)
            name = programs[invocation]
            if name == "hold":
                # Value computed by the first program must survive the repack.
                write(args[1], bytes(b ^ 0x5A for b in read(args[1])))
            elif name == "consume":
                bias, weights, output, x, _code = args
                data = bytes((a + b + c) % 256 for a, b, c in zip(
                    read(x)[:8624], read(weights)[:8624], read(bias) * 7
                ))
                write(output, data)
    expected = bytes(
        ((i * 13 + 58) % 256 + ((i * 13 + 29) % 256 ^ 0x5A)
        + ((i % bias_size) * 13) % 256) % 256 for i in range(8624)
    )
    assert result == expected
    assert not running
    assert all(buffers[v][1] is None for v in live), "LRAM allocation leaked"
    computed = bytes(((i * 13 + 29) % 256) ^ 0x5A for i in range(185856))
    assert computed in stores and computed in reloads, "computed value was not spilled/reloaded"
    return code_load_addresses, code_start_addresses


@pytest.mark.ci
@pytest.mark.parametrize("hw", ["SL2610", "1:256:1:coral_v2:"])
@pytest.mark.parametrize("bias", [1408, 8000])
def test_fragmented_sr200_repack_preserves_values(torq_compiler, tmp_path, bias, hw):
    # Exact independent snapshot arithmetic, not an impossibility inferred from a raw live set.
    pinned = [(0, 1408), (14640, 5120), (66384, 8624), (75008, 185856)]
    ends = [0] + [offset + size for offset, size in pinned]
    starts = [offset for offset, _ in pinned] + [260864]
    assert 260864 - sum(size for _, size in pinned) == 59856
    assert max(start - end for start, end in zip(starts, ends)) == 46624 < 51744
    # With an 8000-byte bias, the hole below code is also too small for output:
    # changing placement priority alone cannot fit; code itself must move.
    ir = fragmented_sr200_ir(bias=bias)
    plain = assign_lram_addresses(torq_compiler, tmp_path, "fragmented-plain", ir,
            "replan=false lram-size=262144", hw=hw)
    assert plain.returncode != 0
    assert "51744-byte buffer" in plain.stderr
    assert f"{201008 + bias - 1408} are pinned" in plain.stderr
    recovered = assign_lram_addresses(torq_compiler, tmp_path, "fragmented-repacked", ir,
      "replan=true lram-size=262144", hw=hw)
    assert recovered.returncode == 0, recovered.stderr
    code_loads, code_starts = check_repacked_values_and_lifetimes(recovered.stdout, bias_size=bias)
    if bias == 8000:
        # This tighter geometry requires moving code, not just changing placement
        # priority. Verify the emitted move without relying on debug-only flags:
        # load code once, relocate it, then start with its preserved contents.
        assert len(code_loads["consume"]) == 1, "consumer code was loaded again from its source"
        assert code_starts["consume"] != code_loads["consume"][0], "consumer code did not move"


@pytest.mark.ci
@pytest.mark.parametrize("mutation", ["missing", "wrong"])
def test_repack_checker_rejects_invalid_program_code(torq_compiler, tmp_path, mutation):
    proc = assign_lram_addresses(torq_compiler, tmp_path, "code-checker",
      fragmented_sr200_ir(bias=8000), "replan=true lram-size=262144")
    assert proc.returncode == 0, proc.stderr
    check_repacked_values_and_lifetimes(proc.stdout, bias_size=8000)
    sources = {}
    for line in proc.stdout.splitlines():
        if "torq_hl.create_invocation" in line:
            name = re.search(r'create_invocation "([^"]+)"', line)[1]
            sources[name] = re.findall(r"%[\w]+", line)[1]
    lines = []
    changed = 0
    for line in proc.stdout.splitlines():
        values = re.findall(r"%[\w]+", line)
        if "torq_hl.load " in line and values[0] == sources["consume"]:
            changed += 1
            if mutation == "missing":
                continue
            # Scratch and consume have equal-length names: the old checker used
            # identical bytes for them and could not distinguish this mistake.
            line = line.replace(sources["consume"], sources["scratch"], 1)
        lines.append(line)
    assert changed == 1
    with pytest.raises(AssertionError, match="program code mismatch: consume"):
        check_repacked_values_and_lifetimes("\n".join(lines), bias_size=8000)


@pytest.mark.ci
def test_pinned_byte_pressure_is_not_repacked(torq_compiler, tmp_path):
    proc = assign_lram_addresses(torq_compiler, tmp_path, "pinned-overflow",
      fragmented_sr200_ir(request=60000),
      "replan=true lram-size=262144")
    assert proc.returncode != 0
    assert "60000-byte buffer" in proc.stderr and "201008 are pinned" in proc.stderr
    assert "byte pressure" in proc.stderr


@pytest.mark.ci
def test_repack_does_not_evict_in_flight_code(torq_compiler, tmp_path):
    proc = assign_lram_addresses(torq_compiler, tmp_path, "still-running",
      fragmented_sr200_ir(background=True),
      "replan=true lram-size=262144")
    assert proc.returncode != 0
    assert "fragmentation" in proc.stderr
    # Replanning changes the failing request from x (51744 bytes) to output
    # (8624 bytes), with 249248 bytes pinned rather than the original 206128.
    assert "8624-byte buffer" in proc.stderr and "249248 are pinned" in proc.stderr
