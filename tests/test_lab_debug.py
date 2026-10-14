# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Classifier and diagnose() tests using captured compiler diagnostics."""

import json
import os

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from _lab_debug_captures import CAPTURES
from torq.lab import LabError
from torq.lab.debug import (
    _BUNDLE_FLAGS,
    Diagnosis,
    _validate_crash_repro,
    _write_bundle,
    classify,
    diagnose,
    normalize_error,
)
from torq.lab.pipeline import tools
from torq.lab.pipeline.workflow import CompileResult

# (fixture, rc, expected class, expected stage, expected normalized message)
# stage/message None = not asserted.
CASES = [
    ("no-npu-lowering-erf-both-disabled.dev.stderr", 1, "no-npu-lowering", "Torq codegen", "error: 'linalg.generic' op unsupported in function body"),
    ("no-npu-lowering-forced-css-disabled.dev.stderr", 1, "no-npu-lowering", "Torq codegen", "error: Cannot execute operation on CSS as disabled by command line option"),
    ("no-npu-lowering-forced-host-disabled.dev.stderr", 1, "no-npu-lowering", "Torq codegen", "error: Cannot execute operation on Host as disabled by command line option"),
    ("memory-stablehlo-gather.dev.stderr", 1, "memory", "Torq codegen", "error: failed to allocate LRAM addresses"),
    # A tensor.empty verifier error alone does not identify memory pressure.
    ("memory-oversized-matmul-tensor-empty.dev.stderr", 1, "unknown", None, "error: 'tensor.empty' op incorrect number of dynamic sizes, has <N>, expected <N>"),
    ("assert-conv-f32-iwidth.dev.stderr", -6, "assert", None, "Assertion `iType == DType::bf<N> && wType == DType::bf<N>' failed."),
    ("assert-conv-fp16-dtype.dev.stderr", -6, "assert", None, 'Assertion `elementType != DType::none && "Invalid element type"\' failed.'),
    ("dynamic-shape-erf.stderr", 1, "dynamic-shape", "Static-shape gate", None),
    ("frontend-unsupported-op-lppool.dev.stderr", 1, "frontend-unsupported-op", "Frontend lowering", "error: failed to legalize operation 'torch.operator' that was explicitly marked illegal"),
    ("assert-matmul-nss-fp32.dev.stderr", -6, "assert", None, 'Assertion `false && "Invalid weight memory format conversion"\' failed.'),
    # rc 1 (not -6): --mlir-pass-pipeline-crash-reproducer flips the assert
    # rc; the classifier must stay text-driven.
    ("assert-matmul-nss-fp32-with-crash-reproducer.dev.stderr", 1, "assert", None, 'Assertion `false && "Invalid weight memory format conversion"\' failed.'),
    ("invalid-model-erf-corrupt.stderr", 1, "invalid-model", "Model validation", None),
    ("import-failed-canned.stderr", 1, "import-failed", "ONNX/TFLite import", None),
    ("environment-missing-css-config-progressive.dev.stderr", 1, "environment", None, None),
    ("timeout-canned.stderr", None, "timeout", None, None),
    ("parse-error-stale-tosa-fc.stderr", 1, "invalid-source", "input-parse", 'error: "-":<N>:<N>: custom op \'tosa.fully_connected\' is unknown'),
    ("parse-error-stale-tosa-dwconv.stderr", 1, "invalid-source", "input-parse", 'error: "-":<N>:<N>: \'tosa.depthwise_conv<N>d\' op expected <N> operands, but found <N>'),
    # Verifier errors are distinct from input parse errors.
    ("verify-error-stale-tosa-collapse-shape.stderr", 1, "unknown", None, "error: 'tensor.collapse_shape' op expected dimension <N> of collapsed type to be static value of <N>"),
    # An assertion takes precedence over the preceding conversion error.
    ("input-conversion-reciprocal-i8-assert.dev.stderr", -6, "assert", None, "error: expected floating point type"),
    # Synthetic release-shape twin of the capture above: what an
    # assertion-compiled-out build shows (diagnostic, rc 1, no assert line).
    ("input-conversion-expected-float-type.release.stderr", 1, "input-conversion", "Input conversion", "error: expected floating point type"),
    # TFLite import: missing [tflite] extra (converter not on PATH).
    ("tflite-import-missing-tool.stderr", 1, "environment", None, None),
    # TFLite import: the converter subprocess exited non-zero.
    ("tflite-import-converter-failed.stderr", 1, "import-failed", "ONNX/TFLite import", None),
]


@pytest.mark.parametrize(
    "fixture,rc,cls,stage,message",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_fixture(fixture, rc, cls, stage, message):
    text = CAPTURES[fixture]
    diag = classify(text, rc)
    assert diag.cls == cls
    if stage is not None:
        assert diag.stage == stage
    assert diag.message == message
    assert diag.doc == f"https://synaptics-torq.github.io/torq-compiler/v/latest/user-manual/debugging_compile_errors.html#{diag.cls}"
    assert diag.hint


# normalize_error units


def test_normalize_error_digits_to_n():
    text = "a.mlir:12:34: error: 'x' op expected 2 operands, but found 1"
    assert normalize_error(text) == "error: 'x' op expected <N> operands, but found <N>"


def test_normalize_error_strips_tile_and_fuse_suffix():
    text = (
        "b.mlir:5:10: error: Cannot execute operation on CSS as disabled by command line "
        "option (encountered while running the pipeline to checking if a tile fits in memory)"
    )
    assert (
        normalize_error(text)
        == "error: Cannot execute operation on CSS as disabled by command line option"
    )


def test_normalize_error_prefers_first_line():
    text = "c.mlir:1:1: error: first failure\nd.mlir:2:2: error: second failure"
    assert normalize_error(text) == "error: first failure"


def test_normalize_error_no_match():
    assert normalize_error("all fine here\n") is None
    assert normalize_error("") is None


# issues.py parity (normalization moved into torq.lab)


class _FakeReport:
    def __init__(self, capstderr=None, longrepr=None):
        self.capstderr = capstderr
        self.longrepr = longrepr


def test_issues_longrepr_fallback_unchanged():
    # torq.testing ships with the repo but not the wheel; skip in the
    # packaged-wheel CI lane where it is unavailable.
    issues = pytest.importorskip(
        "torq.testing.issues", reason="torq.testing not installed (wheel packaging)"
    )
    diagnostic = "model.mlir:4:10: error: bad op 42"
    assert issues._normalize_error(_FakeReport(diagnostic)) == normalize_error(diagnostic)
    assert issues._normalize_error(_FakeReport(None)) == "generic error"
    canned = "no error line here\n"
    assert issues._normalize_error(_FakeReport(canned)) == "generic error"


# Classification precedence

def test_text_matching_two_rules_classifies_as_earlier():
    # no-npu-lowering op error + the chained serialization wrapper: the more
    # specific (earlier) rule must win.
    text = (
        "error: 'linalg.generic' op unsupported in function body\n"
        "error: failed to serialize to torq binary\n"
    )
    assert classify(text, 1).cls == "no-npu-lowering"
    # Without the op error, the same wrapper is plain serialization.
    assert classify("error: failed to serialize to torq binary\n", 1).cls == "serialization"


# rc contract


def test_timeout_is_rc_driven():
    assert classify("anything at all\n", None).cls == "timeout"


def test_crash_negative_rc_without_error_text():
    assert classify("Segmentation fault (core dumped)\n", -11).cls == "crash"


def test_negative_rc_with_unrecognized_error_text_stays_unknown():
    text = "m.mlir:1:1: error: something nobody has seen before"
    assert classify(text, -6).cls == "unknown"
    assert normalize_error(text) in classify(text, -6).excerpt


def test_silent_pass_failure_wrapper_only():
    text = (
        "Compile failed: Error invoking torq-compile (code 1)\n"
        "Diagnostics:\n"
        "work.mlir:2:3: error: failed to run translation of source executable to "
        'target executable for backend #hal.executable.target<"torq", "torq-fb">\n'
        "// -----// IR Dump After TileAndFusePass Failed (torq-tile-and-fuse) //----- //\n"
    )
    diag = classify(text, 1)
    assert diag.cls == "silent-pass-failure"
    assert diag.pass_name == "torq-tile-and-fuse"
    # A second, non-wrapper error line disqualifies it.
    other = text + "work.mlir:9:9: error: another real diagnostic\n"
    assert classify(other, 1).cls == "unknown"


# pass / op / loc extraction


def test_pass_from_crash_reproducer_note_beats_ir_dump():
    text = CAPTURES["assert-matmul-nss-fp32-with-crash-reproducer.dev.stderr"]
    diag = classify(text, 1)
    # The note names the inner failing pass; the IR dumps show only outer ones.
    assert diag.pass_name == "ConvertSliceProgramToTorqHw"
    assert diag.source_loc == "<<SRC>>/compiler/torq/Utils/Kernel.cpp:3353"


def test_pass_from_first_ir_dump():
    text = CAPTURES["frontend-unsupported-op-lppool.dev.stderr"]
    diag = classify(text, 1)
    assert diag.pass_name == "convert-torch-onnx-to-torch"
    assert diag.op == "torch.operator"


def test_pass_from_progress_log_last_open_plus():
    text = (
        "[TORQ dispatch_0] + SomePass\n"
        "[TORQ dispatch_0] - SomePass (1 ms, 0 MB)\n"
        "[TORQ dispatch_0] + FailingPass\n"
    )
    assert classify(text, 1).cls == "unknown"
    assert classify(text, 1).pass_name == "FailingPass"


def test_pre_pipeline_classes_suppress_progress_pass():
    """Setup progress must not be reported as the failing pass."""
    text = CAPTURES["environment-missing-css-config-progressive.dev.stderr"]
    diag = classify(text, 1)
    assert diag.cls == "environment"
    assert diag.pass_name is None  # '+ TileAndFuse' is setup noise
    assert diag.op is None
    assert diag.source_loc is None
    assert diag.excerpt == "Unable to find css config no_such_css_config"


def test_pre_pipeline_marker_excerpt_when_no_error_line():
    # dynamic-shape keeps the marker-line excerpt.
    diag = classify(CAPTURES["dynamic-shape-erf.stderr"], 1)
    assert diag.pass_name is None
    assert diag.excerpt  # the marker line, not generic noise
    # invalid-model / import-failed carry the full LabError text: the
    # node-level checker message / importer traceback is the content.
    for fixture in ("invalid-model-erf-corrupt.stderr", "import-failed-canned.stderr"):
        text = CAPTURES[fixture]
        diag = classify(text, 1)
        assert diag.pass_name is None
        assert diag.excerpt == "\n".join(text.strip().splitlines()[-30:])
    assert "input 'ghost' of node" in classify(CAPTURES["invalid-model-erf-corrupt.stderr"], 1).excerpt
    assert "OnnxImportError" in classify(CAPTURES["import-failed-canned.stderr"], 1).excerpt


def test_invalid_model_excerpt_capped_at_last_30_lines():
    lines = [f"filler line {i}" for i in range(40)]
    text = "\n".join(lines) + "\nONNX model validation failed for m.onnx\n"
    diag = classify(text, 1)
    assert diag.cls == "invalid-model"
    assert len(diag.excerpt.splitlines()) == 30
    assert diag.excerpt.splitlines()[-1] == "ONNX model validation failed for m.onnx"


def test_unknown_excerpt_carries_raw_tail():
    text = CAPTURES["verify-error-stale-tosa-collapse-shape.stderr"]
    diag = classify(text, 1)
    assert diag.cls == "unknown"
    assert diag.message in diag.excerpt
    tail = "\n".join(text.splitlines()[-10:])
    assert tail in diag.excerpt


# raw_message: verbatim first error line


def test_format_diagnosis_prefers_raw_message():
    from torq.lab.cli.output import format_diagnosis

    d = Diagnosis(
        cls="assert", hint="h",
        message="Assertion `iType == DType::bf<N> && wType == DType::bf<N>' failed.",
        raw_message="Assertion `iType == DType::bf16 && wType == DType::bf16' failed.",
    )
    out = format_diagnosis(d)
    assert "  message: Assertion `iType == DType::bf16 && wType == DType::bf16' failed." in out
    assert "bf<N>" not in out


# known asserts


def test_known_assert_hint_replaces_report_hint():
    text = CAPTURES["assert-conv-f32-iwidth.dev.stderr"]
    diag = classify(text, -6)
    assert diag.cls == "assert"
    assert diag.hint == (
        "Known-unsupported configuration, not a compiler bug: f32 conv requires "
        "bf16 input/weights (convert the model dtype, e.g. torq-lab convert_dtype, and recompile)"
    )
    assert diag.source_loc == "<<SRC>>/compiler/torq/Utils/Kernel.cpp:3604"
    assert "bf16" in diag.raw_message
    assert "bf<N>" in diag.message
    # A different assert keeps the generic report hint.
    other = classify(CAPTURES["assert-matmul-nss-fp32.dev.stderr"], -6)
    assert other.hint == "Compiler bug: quote the assert file:line from the message and report with the crash backtrace."


# op_source resolver


class _LocPipe:
    def __init__(self, work_dir, model_path):
        self.work_dir = work_dir
        self.config = type("C", (), {"model_path": model_path})()


LPP_IMPORTED_MLIR = """\
module {
  func.func @lpp(%arg0: !torch.vtensor<[1,1,16,16],f32>) -> !torch.vtensor<[1,1,14,14],f32> {
    %0 = torch.operator "onnx.LpPool"(%arg0) {torch.onnx.kernel_shape = [3 : si64, 3 : si64], torch.onnx.p = 2 : si64} : (!torch.vtensor<[1,1,16,16],f32>) -> !torch.vtensor<[1,1,14,14],f32>
    return %0 : !torch.vtensor<[1,1,14,14],f32>
  }
}
"""


def test_source_op_resolves_imported_line(tmp_path):
    from torq.lab.debug import _source_node, _source_op

    (tmp_path / "lppool.mlir").write_text(LPP_IMPORTED_MLIR)
    pipe = _LocPipe(tmp_path, tmp_path / "lppool.onnx")
    assert _source_op(pipe, "lppool.mlir:3:5") == "onnx.LpPool"
    # _source_node keeps its `onnx.<Op> at file:line:col` form for op_summary.
    assert _source_node(pipe, "lppool.mlir:3:5") == "onnx.LpPool at lppool.mlir:3:5"
    # Past EOF and a non-torch line yield None.
    assert _source_op(pipe, "lppool.mlir:9:9") is None
    assert _source_op(pipe, "lppool.mlir:2:3") is None
    mlir_pipe = _LocPipe(tmp_path, tmp_path / "model.mlir")
    assert _source_op(mlir_pipe, "model.mlir:3:5") is None


def test_format_diagnosis_prints_source_op():
    from torq.lab.cli.output import format_diagnosis

    d = Diagnosis(
        cls="frontend-unsupported-op", op="torch.operator", source_loc="lppool.mlir:4:10",
        message="error: x", op_source="onnx.LpPool", hint="h",
    )
    out = format_diagnosis(d)
    assert "  source op: onnx.LpPool" in out
    assert out.index("location:") < out.index("source op:")


# bisect output


def test_format_diagnosis_bisect_all_pass():
    from torq.lab.cli.output import format_diagnosis

    d = Diagnosis(cls="memory", hint="h")
    d.bisect = [{"layer": f"L{i}", "cls": "ok", "message": None} for i in range(3)]
    out = format_diagnosis(d)
    assert "every layer passed but the model failed: the cause is fusion, tiling, or memory" in out


# diagnose() with fake pipes


class _FailingPipe:
    """Record compiler options; fail with the configured ToolError unless a pass flag is set."""

    def __init__(self, tmp_path, stderr, rc=1, user_opts=(), model_name="model.mlir", pass_flags=()):
        self.work_dir = tmp_path / "work"
        self.work_dir.mkdir(parents=True)
        self.debug_dir = self.work_dir / "debug"
        self.config = type(
            "C",
            (),
            {
                "model_path": self.work_dir / model_name,
                "compiler_options": list(user_opts),
                "chip": "SL2610",
                "runtime_hw_type": "sim",
                "timeout": None,
            },
        )()
        self._stderr, self._rc, self._pass_flags = stderr, rc, pass_flags
        self.calls = []

    def _ensure_work_dir(self):
        pass

    def compile(self):
        opts = list(self.config.compiler_options)
        self.calls.append(opts)
        cmd = ["fake", str(self.config.model_path), *opts]
        if any(f in opts for f in self._pass_flags):
            return CompileResult(vmfb_path=self.work_dir / "m.vmfb", command=cmd, elapsed=0.0, diagnostics="")
        raise tools.ToolError(cmd, self._rc, self._stderr)


# Real imported MLIR (torch-mlir ONNX import of a one-op Erf; line 4 carries
# the torch.operator node line). Node names are lost in import by design.
ERF_IMPORTED_MLIR = """\
module {
  func.func @erf(%arg0: !torch.vtensor<[1,64],f32>) -> !torch.vtensor<[1,64],f32> attributes {torch.onnx_meta.ir_version = 13 : si64, torch.onnx_meta.opset_version = 13 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Erf"(%arg0) : (!torch.vtensor<[1,64],f32>) -> !torch.vtensor<[1,64],f32>
    return %0 : !torch.vtensor<[1,64],f32>
  }
}
"""

# Real failed-pass IR block (torq-compile --mlir-print-ir-after-failure,
# SerializeTargetExecutablesPass on the erf model); @WORK@ is the work dir so
# the callsite locs point at the imported MLIR the test writes there.
ERF_FAILED_PASS_BLOCK = """\
// -----// IR Dump After SerializeTargetExecutablesPass Failed (iree-hal-serialize-target-executables) //----- //
hal.executable private @erf_dispatch_0 {
  hal.executable.variant public @torq_fb target(<"torq", "torq-fb">) {
    builtin.module {
      func.func @erf_dispatch_0_elementwise_1x64_f32() {
        %0 = "torq_hl.map_binding"() : () -> memref<1x64xf32> loc("@WORK@/erf_f32.mlir":2:3)
        %1 = "torq_hl.map_binding"() : () -> memref<1x64xf32> loc("@WORK@/erf_f32.mlir":2:3)
        linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%0 : memref<1x64xf32>) outs(%1 : memref<1x64xf32>) attrs =  {"torq-action-id" = 0 : i32} {
        ^bb0(%in: f32 loc(callsite("@WORK@/erf_f32.mlir":4:10 at "@WORK@/erf_f32.mlir":2:3)), %out: f32 loc(callsite("@WORK@/erf_f32.mlir":4:10 at "@WORK@/erf_f32.mlir":2:3))):
          %2 = math.erf %in : f32 loc(callsite("@WORK@/erf_f32.mlir":4:10 at "@WORK@/erf_f32.mlir":2:3))
          linalg.yield %2 : f32 loc(callsite("@WORK@/erf_f32.mlir":4:10 at "@WORK@/erf_f32.mlir":2:3))
        } loc(callsite("@WORK@/erf_f32.mlir":4:10 at "@WORK@/erf_f32.mlir":2:3))
        return loc("@WORK@/erf_f32.mlir":2:3)
      }
    }
  }
} loc("@WORK@/erf_f32.mlir":2:3)

"""

MATCHER_TRACE = """\
[dialect-conversion:1]   * Pattern : 'linalg.generic -> ()' {
[dialect-conversion:1]     ** Failure : not a sigmoid generic
[dialect-conversion:1]   } -> FAILURE : pattern failed to match
[dialect-conversion:1]   * Pattern : 'linalg.generic -> ()' {
[dialect-conversion:1]     ** Failure : not an elementwise add
[dialect-conversion:1]   } -> FAILURE : pattern failed to match
"""


class _ErfPipe(_FailingPipe):
    """The Erf no-npu-lowering failure, with passing fallback probes and a matcher trace."""

    def __init__(self, tmp_path, capability=True):
        stderr = ERF_FAILED_PASS_BLOCK.replace("@WORK@", str(tmp_path / "work")) + CAPTURES[
            "no-npu-lowering-erf-both-disabled.dev.stderr"
        ]
        super().__init__(tmp_path, stderr, user_opts=("--torq-disable-css", "--torq-disable-host"),
                         model_name="erf_f32.onnx")
        (self.work_dir / "erf_f32.mlir").write_text(ERF_IMPORTED_MLIR)
        self._capability = capability

    def debug_only_available(self):
        return self._capability

    def compile(self):
        opts = list(self.config.compiler_options)
        self.calls.append(opts)
        if "--mlir-print-ir-after=torq-assign-operations-to-cpu-programs" in opts:
            # A fallback-probe re-run: it passes, and the assign-pass IR
            # names the executor that took the op.
            program = "css_linalg.generic_0" if "--torq-disable-host" in opts else "host_linalg.generic_0"
            return CompileResult(
                vmfb_path=self.work_dir / "erf_f32.vmfb",
                command=["fake", "erf_f32.mlir"],
                elapsed=0.0,
                diagnostics=f'    %2 = torq_hl.program "{program}" : !torq_hl.program<...> {{\n',
            )
        if any(o.startswith("--debug-only=") for o in opts):
            # The matcher-reason re-run: same failure, with the trace.
            raise tools.ToolError(["fake", "erf_f32.mlir"], 1, self._stderr + MATCHER_TRACE)
        raise tools.ToolError(["fake", "erf_f32.mlir"], 1, self._stderr)


def test_diagnose_no_npu_lowering_op_summary(tmp_path):
    pipe = _ErfPipe(tmp_path)
    diag = diagnose(pipe)
    assert diag.cls == "no-npu-lowering"
    s = diag.op_summary
    assert s["op"] == "linalg.generic"
    assert s["indexing_maps"] == [
        "affine_map<(d0, d1) -> (d0, d1)>",
        "affine_map<(d0, d1) -> (d0, d1)>",
    ]
    assert s["iterator_types"] == ["parallel", "parallel"]
    assert s["operand_element_types"] == ["f32"]
    assert s["result_element_types"] == ["f32"]
    assert s["body_ops"] == ["math.erf"]
    assert s["loc"].endswith("erf_f32.mlir:4:10")
    assert s["source"] == "onnx.Erf at erf_f32.mlir:4:10"
    assert s["nearest_supported"]["ops_md_section"] == "Element-Wise Operators"
    assert s["fallback"] == {
        "--torq-disable-css": {"passed": True, "executor": "css"},
        "--torq-disable-host": {"passed": True, "executor": "host"},
    }
    assert diag.probes == {"--torq-disable-css": True, "--torq-disable-host": True}
    assert "not a sigmoid generic" in s["matcher_reasons"]
    assert "not an elementwise add" in s["matcher_reasons"]
    assert s["debug_only_available"] is True
    # The repro bundle, with failed-pass.mlir from the real dump block.
    failed = pipe.debug_dir / "failed-pass.mlir"
    assert failed.exists()
    assert "IR Dump After SerializeTargetExecutablesPass Failed" in failed.read_text()
    assert "linalg.generic" in failed.read_text()
    bundle = json.loads((pipe.debug_dir / "diagnosis.json").read_text())
    assert bundle["op_summary"]["source"] == "onnx.Erf at erf_f32.mlir:4:10"
    assert str(failed) in diag.repro
    # Verdict compile (stage 1) + bundle compile (stage 2) + two fallback
    # probes + one matcher-reason run.
    assert len(pipe.calls) == 5


def test_diagnose_no_npu_lowering_capability_unavailable(tmp_path):
    pipe = _ErfPipe(tmp_path, capability=False)
    diag = diagnose(pipe)
    s = diag.op_summary
    assert s["matcher_reasons"] == ["matcher reasons need an assert-enabled build"]
    assert s["debug_only_available"] is False
    # No matcher re-run when the capability probe fails.
    assert all(
        not any(o.startswith("--debug-only=") for o in opts)
        for opts in pipe.calls
    )


def test_diagnose_laberror_from_import_in_compile(tmp_path):
    # An import/validation LabError surfaces from compile() (the real
    # ModelPipeline.compile calls ensure_mlir internally); diagnose classifies
    # it from the message without any tool command having run.
    text = CAPTURES["invalid-model-erf-corrupt.stderr"]

    class _BrokenImportPipe(_FailingPipe):
        def compile(self):
            raise LabError(text)

    pipe = _BrokenImportPipe(tmp_path, text)
    diag = diagnose(pipe)
    assert diag.cls == "invalid-model"
    assert (pipe.debug_dir / "diagnosis.json").exists()
    assert not (pipe.debug_dir / "repro.sh").exists()  # no tool command ran
    assert diag.repro


# Two-stage diagnosis: verdict and reproduction bundle


def test_two_stage_verdict_and_bundle(tmp_path):
    stderr = CAPTURES["frontend-unsupported-op-lppool.dev.stderr"]  # non-memory, rc 1
    pipe = _FailingPipe(tmp_path, stderr, user_opts=("--mlir-elide-elementsattrs-if-larger=8",))
    diag = diagnose(pipe)
    assert diag.cls == "frontend-unsupported-op"
    stage1, stage2 = pipe.calls
    # Stage 1 (verdict): the user's options plus display flags; the user's
    # =-form elision wins over the display variant.
    assert stage1 == ["--mlir-elide-elementsattrs-if-larger=8", "--torq-show-pass-progress"]
    # Stage 2 (bundle): the bundle set with the crash reproducer, no elision.
    debug = pipe.debug_dir
    assert stage2 == [*_BUNDLE_FLAGS, f"--mlir-pass-pipeline-crash-reproducer={debug / 'crash-repro.mlir'}"]
    # raw-stderr.txt is the stage-1 full stderr.
    raw = debug / "raw-stderr.txt"
    assert raw.read_text() == stderr
    data = json.loads((debug / "diagnosis.json").read_text())
    assert data["cls"] == "frontend-unsupported-op"
    assert data["raw_stderr"] == str(raw)
    assert data["repro"] == [str(debug / n) for n in ("diagnosis.json", "repro.sh", "raw-stderr.txt")]
    assert data["metadata"]["chip"] == "SL2610"
    assert {"tool", "tool_version", "runtime_hw_type", "platform", "machine", "created"} <= data["metadata"].keys()
    repro = debug / "repro.sh"
    assert os.access(repro, os.X_OK)
    text = repro.read_text()
    assert "# 1. the user's failing compile (expected exit code 1)" in text
    assert " ".join(["fake", str(pipe.config.model_path), *stage1]) in text


def test_stage1_timeout_single_invocation_names_pass(tmp_path):
    stderr = (
        "[TORQ dispatch_0] + TorqPassA\n"
        "[TORQ dispatch_0] - TorqPassA (1 ms, 0 MB)\n"
        "[TORQ dispatch_0] + SpinnyPass\n"
        "Compile failed: Error invoking torq-compile (timeout)\n"
    )
    pipe = _FailingPipe(tmp_path, stderr, rc=None)
    diag = diagnose(pipe, probe=True)
    assert diag.cls == "timeout"
    assert diag.pass_name == "SpinnyPass"  # progress-based, from stage 1
    assert len(pipe.calls) == 1  # no stage-2 bundle run after a timeout
    assert diag.probes == {}
    assert "the pass above is the last one started before the timeout" in diag.hint
    assert "# 1. the user's failing compile (expected to time out)" in (pipe.debug_dir / "repro.sh").read_text()


def test_repro_sh_crash_repro_label_and_validation(tmp_path):
    stderr = CAPTURES["assert-matmul-nss-fp32-with-crash-reproducer.dev.stderr"]
    pipe = _FailingPipe(tmp_path, stderr, rc=-6)
    pipe.debug_dir.mkdir(parents=True, exist_ok=True)
    (pipe.debug_dir / "raw-stderr.txt").write_text(stderr)
    (pipe.debug_dir / "crash-repro.mlir").write_text("module {\n}\n")
    cmd = ["fake-tool", "model.mlir", "-o", "out.vmfb", "--torq-hw=SL2610",
           "--torq-executor-map=m.json", "--torq-show-pass-progress",
           "--mlir-elide-elementsattrs-if-larger=16"]
    diag = Diagnosis(cls="assert", pass_name="ConvertSliceProgramToTorqHw", hint="h")
    diag.raw_stderr = str(pipe.debug_dir / "raw-stderr.txt")
    paths = _write_bundle(pipe, diag, cmd, -6)
    text = (pipe.debug_dir / "repro.sh").read_text()
    assert "set -e" not in text
    # A negative rc is a signal; the label carries the shell's exit code.
    assert "# 1. the user's failing compile (expected to crash (shell exit code 134))" in text
    assert "# 2. standalone crash reproducer" in text
    assert "/tmp/crash-repro.vmfb" in text
    # Command 2 compiles the reproducer, not the model, and drops the
    # stage-1 display flags (elision would corrupt the reproducer).
    assert "model.mlir" not in text.split("# 2.")[1]
    assert "--mlir-elide-elementsattrs-if-larger" not in text.split("# 2.")[1]
    assert "--torq-show-pass-progress" not in text.split("# 2.")[1]
    assert "--torq-executor-map=m.json" in text.split("# 2.")[1]
    for name in ("crash-repro.mlir", "raw-stderr.txt", "repro.sh", "diagnosis.json"):
        assert str(pipe.debug_dir / name) in paths
    data = json.loads((pipe.debug_dir / "diagnosis.json").read_text())
    # The bare module parses when the IREE bindings are present; without
    # them the status degrades instead of failing the command.
    assert data["metadata"]["crash_repro_parse"] in ("ok", "unavailable")


def test_crash_repro_validation_elided(tmp_path):
    p = tmp_path / "crash-repro.mlir"
    p.write_text("%cst_0 = arith.constant dense_resource<__elided__> : tensor<8x1x1x32xi8>\n")
    assert _validate_crash_repro(p) == "failed: elided content detected"


# probe: the differential table


def test_probe_memory_flags(tmp_path):
    from torq.lab.cli.output import format_diagnosis

    user = "--torq-max-nss-programs-size=8388608"
    pipe = _FailingPipe(
        tmp_path, CAPTURES["memory-stablehlo-gather.dev.stderr"], user_opts=(user,),
        pass_flags=("--torq-max-nss-programs-size=16777216",),
    )
    diag = diagnose(pipe, probe=True)
    assert diag.probes == {
        "--torq-max-nss-programs-size=16777216": True,
    }
    # Memory skips the stage-2 bundle run; the probe re-runs are the user's
    # options plus one probe flag (replacing the user's =-form variant) and
    # none of the diagnostic flags.
    assert pipe.calls[1:] == [["--torq-max-nss-programs-size=16777216"]]
    assert pipe.config.compiler_options == [user]  # restored after diagnose
    assert diag.hint == "Memory pressure: try --torq-max-nss-programs-size."
    assert "the default 8 MB NSS program reservation was the limit; 16 MB fits" in format_diagnosis(diag)


def test_probe_skips_a_flag_the_user_already_passed(tmp_path):
    stderr = CAPTURES["verify-error-stale-tosa-collapse-shape.stderr"]
    pipe = _FailingPipe(tmp_path, stderr, rc=1, user_opts=("--torq-disable-slices",))
    diag = diagnose(pipe, probe=True)
    assert diag.cls == "unknown"
    assert diag.probes == {}
    # Stage 1 + the stage-2 bundle run; the probe re-run is still skipped
    # because it would repeat the failed command.
    assert len(pipe.calls) == 2


def test_input_conversion_probe_flag(tmp_path):
    stderr = CAPTURES["input-conversion-expected-float-type.release.stderr"]
    pipe = _FailingPipe(tmp_path, stderr, rc=1)
    diag = diagnose(pipe, probe=True)
    assert diag.cls == "input-conversion"
    assert diag.probes == {"--torq-disable-dispatch-fusion": False}


# bisect: per-layer ONNX compiles


def test_bisect_classifies_each_onnx_layer(tmp_path, monkeypatch):
    import torq.lab.debug as debug_mod

    pipe = _FailingPipe(tmp_path, CAPTURES["memory-stablehlo-gather.dev.stderr"], model_name="small.onnx")
    # erf -> div -> softmax, one ONNX node each.
    w = numpy_helper.from_array(np.ones((1, 64), dtype=np.float32), "divisor")
    nodes = [
        helper.make_node("Erf", ["x"], ["t1"], name="erf"),
        helper.make_node("Div", ["t1", "divisor"], ["t2"], name="div"),
        helper.make_node("Softmax", ["t2"], ["y"], name="softmax", axis=-1),
    ]
    graph = helper.make_graph(
        nodes, "small",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 64])],
        initializer=[w],
    )
    onnx.save(helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]),
              str(pipe.config.model_path))

    class _StubPipeline:
        """The Erf layer compiles; every other layer asserts."""

        def __init__(self, config, remote=None):
            self.config = config

        def compile(self):
            if "Erf" not in self.config.model_path.name:
                raise tools.ToolError(["fake"], -6, CAPTURES["assert-conv-f32-iwidth.dev.stderr"])

    monkeypatch.setattr(debug_mod, "ModelPipeline", _StubPipeline)
    diag = diagnose(pipe, bisect=True)
    assert diag.cls == "memory"
    assert [(e["layer"], e["cls"]) for e in diag.bisect] == [
        ("layer_Erf_0", "ok"), ("layer_Div_1", "assert"), ("layer_Softmax_2", "assert"),
    ]
    assert diag.bisect[0]["message"] is None
    assert diag.bisect[1]["message"].startswith("Assertion")


def test_bisect_checks_all_layers_in_large_model(tmp_path, monkeypatch):
    import torq.lab.debug as debug_mod
    from torq.lab.model_tools.extraction.onnx import layers as onnx_layers

    # A real 101-node ONNX file (get_full_model loads it), but the
    # extraction and the per-layer compiles are stubbed.
    nodes, prev = [], "x"
    for i in range(101):
        nxt = "y" if i == 100 else f"t{i}"
        nodes.append(helper.make_node("Erf", [prev], [nxt], name=f"erf{i}"))
        prev = nxt
    big = helper.make_graph(
        nodes, "big",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 64])],
    )
    one = helper.make_graph(
        [helper.make_node("Erf", ["x"], ["y"])], "l",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 64])],
    )
    opset = [helper.make_opsetid("", 13)]
    stderr = CAPTURES["memory-stablehlo-gather.dev.stderr"]
    pipe = _FailingPipe(tmp_path, stderr, rc=1, model_name="big101.onnx")
    onnx.save(helper.make_model(big, opset_imports=opset), str(pipe.config.model_path))
    monkeypatch.setattr(
        onnx_layers, "generate_onnx_layers_from_model",
        lambda model: {f"layer_Erf_{i}": helper.make_model(one, opset_imports=opset)
                       for i in range(101)},
    )

    class _StubPipeline:
        def __init__(self, config, remote=None):
            self.config = config

        def compile(self):
            if self.config.model_path.stem == "layer_Erf_100":
                raise tools.ToolError(["fake"], 1, "error: failed to allocate LRAM addresses\n")

    monkeypatch.setattr(debug_mod, "ModelPipeline", _StubPipeline)
    diag = diagnose(pipe, bisect=True)
    assert diag.cls == "memory"
    assert len(diag.bisect) == 101
    assert all(e["cls"] == "ok" for e in diag.bisect[:-1])
    assert diag.bisect[0]["layer"] == "layer_Erf_0"
    assert diag.bisect[-1]["layer"] == "layer_Erf_100"
    assert diag.bisect[-1]["cls"] == "memory"
    data = json.loads((pipe.debug_dir / "diagnosis.json").read_text())
    assert data["bisect"] == diag.bisect
