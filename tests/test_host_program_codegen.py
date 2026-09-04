import re
import subprocess

import pytest

from torq.testing.iree import MODELS_DIR

UNFUSE_FMA_PASS = "iree-llvmcpu-unfuse-fma-pass"
UNFUSE_FMA_MARKER = "IR Dump After LLVMCPUUnfuseFMAOpsPass"

HOST_MODEL = MODELS_DIR / "tosa_ops_host_css" / "softmax-1x1000xi8.mlir"

# Softmax's exp lowering leaves math.fma on the host; slices are off so the whole
# graph falls back to the CPU programs compiled by CompileCpuProgramsPass.
BASE_OPTIONS = [
    "--iree-hal-target-backends=torq",
    "--torq-hw=SL2610",
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


def _compile(torq_compiler, out, extra_options):
    cmd = [
        str(torq_compiler.file_path),
        str(HOST_MODEL),
        "-o",
        str(out),
        *BASE_OPTIONS,
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
