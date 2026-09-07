import re
import subprocess

import pytest

from torq.testing.iree import MODELS_DIR

UNFUSE_FMA_PASS = "iree-llvmcpu-unfuse-fma-pass"
UNFUSE_FMA_MARKER = "IR Dump After LLVMCPUUnfuseFMAOpsPass"

HOST_MODEL = MODELS_DIR / "tosa_ops_host_css" / "softmax-1x1000xi8.mlir"

# SL2610 uses the coral_v1 CSS config (mabi=ilp32, soft float). The custom hw
# spec is <hw_id>:<lram_size_kb>:<slice_count>:<css_features>:<nss_features>;
# coral_v2 is the hard-float (mabi=ilp32f) CSS config.
SOFT_FLOAT_HW = "--torq-hw=SL2610"
HARD_FLOAT_HW = "--torq-hw=0:512:2:coral_v2:nss_v2"

# Softmax's exp lowering leaves math.fma on the host; slices are off so the whole
# graph falls back to the CPU programs compiled by CompileCpuProgramsPass.
BASE_OPTIONS = [
    "--iree-hal-target-backends=torq",
    "--torq-disable-slices",
    "--torq-target-host-triple=native",
]


def _latest_phase_with(phases_dir, needle):
    """Return the text of the highest-numbered compilation-phase dump that
    contains `needle`, or None. Phase files are named like
    `<model>.<n>.<phase>.mlir`; picking the latest avoids counting the same op
    across several phase dumps."""
    matches = []
    for p in phases_dir.glob("*.mlir"):
        text = p.read_text()
        if needle not in text:
            continue
        m = re.search(r"\.(\d+)\.", p.name)
        matches.append((int(m.group(1)) if m else -1, text))
    if not matches:
        return None
    return max(matches, key=lambda t: t[0])[1]


def _arg_accesses(line):
    """Parse the `arg_accesses = [1 : i32, ...]` bitfields off a start_program line."""
    m = re.search(r"arg_accesses = \[([^\]]*)\]", line)
    if not m:
        return []
    return [int(v) for v in re.findall(r"(\d+)\s*:\s*i32", m.group(1))]


def _compile(torq_compiler, out, extra_options, hw=SOFT_FLOAT_HW):
    cmd = [
        str(torq_compiler.file_path),
        str(HOST_MODEL),
        "-o",
        str(out),
        *BASE_OPTIONS,
        hw,
        *extra_options,
    ]
    print("Compiling with:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"compile failed:\n{result.stderr}"
    return result.stderr


@pytest.mark.ci
def test_host_programs_keep_fused_fma(torq_compiler, tmp_path):
    """Host CPU programs must not run the unfuse-fma pass: it splits a fused fma
    into an unflagged mulf/addf after fast-math flags are set, which stops the
    aarch64 backend from re-fusing them into one NEON instruction
    (synaptics-torq/torq-compiler-dev#2317).

    CSS programs still need the split, so the same model compiled with CSS
    enabled is the control that keeps the pass name and the print flag honest.
    """
    print_options = [
        "--mlir-disable-threading",
        f"--mlir-print-ir-after={UNFUSE_FMA_PASS}",
    ]

    host_only = _compile(
        torq_compiler,
        tmp_path / "host_only.vmfb",
        ["--torq-disable-css", *print_options],
    )
    host_runs = host_only.count(UNFUSE_FMA_MARKER)
    assert host_runs == 0, (
        f"{UNFUSE_FMA_PASS} ran {host_runs} time(s) on the host path; it was "
        "probably re-added to addHostLoweringPasses"
    )

    with_css = _compile(torq_compiler, tmp_path / "with_css.vmfb", print_options)
    css_runs = with_css.count(UNFUSE_FMA_MARKER)
    assert css_runs > 0, (
        f"{UNFUSE_FMA_PASS} ran {css_runs} time(s) with CSS enabled; the pass "
        "was dropped from the CSS path, or the pass/flag name changed and the "
        "host half of this test proves nothing"
    )


@pytest.mark.ci
def test_hard_float_css_programs_keep_fused_fma(torq_compiler, tmp_path):
    """Only soft-float CSS targets need the unfuse-fma pass. coral_v2 is
    mabi=ilp32f and lowers an fma to a single fmadd, so splitting it there costs
    an instruction and buys nothing
    (synaptics-torq/torq-compiler-dev#2317)."""
    phases = tmp_path / "phases"

    hard_float = _compile(
        torq_compiler,
        tmp_path / "hard_float.vmfb",
        [
            "--mlir-disable-threading",
            f"--mlir-print-ir-after={UNFUSE_FMA_PASS}",
            f"--dump-compilation-phases-to={phases}",
        ],
        hw=HARD_FLOAT_HW,
    )

    assert _latest_phase_with(phases, "program<css>"), (
        "the hard-float compile produced no CSS program, so the marker count "
        "below would pass even if the gate were wrong"
    )

    runs = hard_float.count(UNFUSE_FMA_MARKER)
    assert runs == 0, (
        f"{UNFUSE_FMA_PASS} ran {runs} time(s) on a hard-float CSS target; the "
        "soft-float gate in addCssLoweringPasses is gone or reads the wrong mabi"
    )


@pytest.mark.ci
def test_host_start_program_carries_arg_accesses(torq_compiler, tmp_path):
    """Every host torq_hl.start_program must carry an arg_accesses attribute:
    the runtime skips the XRAM write-back of an argument only for the ones the
    compiler marks read-only there, so a missing attribute silently turns into
    extra copies (synaptics-torq/torq-compiler-dev#2317)."""
    phases = tmp_path / "phases"

    _compile(
        torq_compiler,
        tmp_path / "model.vmfb",
        ["--torq-disable-css", f"--dump-compilation-phases-to={phases}"],
    )

    ir = _latest_phase_with(phases, "torq_hl.start_program")
    assert ir, "no compilation phase contained a torq_hl.start_program op"

    start_lines = [ln.strip() for ln in ir.splitlines() if "torq_hl.start_program" in ln]
    assert start_lines, "no torq_hl.start_program op in the latest phase dump"

    missing = [ln for ln in start_lines if "arg_accesses" not in ln]
    assert not missing, (
        f"{len(missing)} of {len(start_lines)} start_program ops have no "
        f"arg_accesses attribute, e.g.:\n{missing[0]}"
    )

    accesses = [_arg_accesses(ln) for ln in start_lines]

    unknown = sorted({v for entry in accesses for v in entry} - {1, 2, 3})
    assert not unknown, (
        f"start_program arg_accesses entries must be Read(1), Write(2) or "
        f"Read|Write(3), got {unknown}"
    )

    # The outliner drops the load of an init the program never reads, and the
    # host fallback creates a fresh tensor.empty init for every output, so the
    # first argument of at least one program must come out Write-only.
    write_only = [entry for entry in accesses if entry and entry[0] == 2]
    assert write_only, (
        "no host start_program has a Write-only init; the accesses recorded on "
        f"torq_hl.program are not reaching start_program:\n"
        + "\n".join(str(entry) for entry in accesses)
    )
