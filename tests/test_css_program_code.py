import re
import subprocess
from pathlib import Path

from torq.testing.iree import MODELS_DIR


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


def test_css_program_code_is_shared_across_invocations(request, torq_compiler, tmp_path):
    """A CSS program invoked multiple times must share a single torq_hl.program_code
    op, so its binary is placed in XRAM exactly once instead of once per invocation
    (synaptics-torq/torq-compiler-dev#1615).

    The depthwise-conv1d model, with linalg slicing enabled, invokes one CSS program
    for several tiles, producing multiple CSS create_invocation ops that should all
    reference the same program_code value and the same XRAM address.
    """
    model = MODELS_DIR / "torch_ops" / "depthwise-conv1d-bf16-c64-w32.mlir"
    phases = tmp_path / "phases"

    cmd = [
        str(torq_compiler.file_path),
        str(model),
        "-o",
        str(tmp_path / "model.vmfb"),
        "--torq-hw=SL2610",
        "--torq-disable-linalg-slicing=false",
        "--torq-target-host-triple=native",
        f"--dump-compilation-phases-to={phases}",
    ]
    print("Compiling with:", " ".join(cmd))
    subprocess.check_call(cmd)

    ir = _latest_phase_with(phases, "torq_hl.program_code")
    assert ir, "no compilation phase contained a torq_hl.program_code op"

    program_code_lines = [ln for ln in ir.splitlines() if "torq_hl.program_code" in ln]
    css_invocations = re.findall(r'create_invocation "css', ir)

    # Exactly one program_code per CSS program (this model has a single CSS program)
    # but multiple CSS invocations -> the code section is shared, not duplicated.
    assert len(program_code_lines) == 1, (
        f"expected one program_code, got {len(program_code_lines)}"
    )
    assert len(css_invocations) > len(program_code_lines), (
        f"CSS code is not shared: {len(css_invocations)} invocations but "
        f"{len(program_code_lines)} program_code op(s)"
    )

    # The shared code section is placed in XRAM exactly once: the single
    # program_code carries one xram_address.
    xram_assigned = [ln for ln in program_code_lines if "xram_address" in ln]
    assert len(xram_assigned) == 1, (
        f"expected exactly one program_code with an xram_address, got {len(xram_assigned)}"
    )

    # A successful compile additionally proves each CSS create_invocation carries
    # only the per-invocation args section: serializeCssInvocation rejects any CSS
    # invocation that still has a code section (it expects exactly one section).


def test_program_code_rejects_non_css_program(torq_compiler, tmp_path):
    """torq_hl.program_code only models CSS programs' position-independent code.
    NSS/Slice bitstreams bake in addresses and are not shareable this way, so the
    op verifier must reject a program_code on a non-CSS program
    (synaptics-torq/torq-compiler-dev#1615)."""
    iree_opt = Path(torq_compiler.file_path).parent / "iree-opt"
    assert iree_opt.exists(), f"iree-opt not found next to torq-compile: {iree_opt}"

    mlir = tmp_path / "nss_program_code.mlir"
    mlir.write_text(
        "func.func @f(%p: !torq_hl.program<nss>) -> memref<4xi8> {\n"
        "  %c = torq_hl.program_code %p : !torq_hl.program<nss> -> memref<4xi8>\n"
        "  return %c : memref<4xi8>\n"
        "}\n"
    )

    result = subprocess.run(
        [str(iree_opt), str(mlir)], capture_output=True, text=True
    )
    assert result.returncode != 0, "verifier accepted program_code on a non-CSS program"
    assert "only CSS programs" in result.stderr, result.stderr
