# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Classify compiler failures and suggest the next debugging step.

See https://synaptics-torq.github.io/torq-compiler/v/latest/user-manual/debugging_compile_errors.html
for the diagnosis classes and triage workflows.
"""

import json
import logging
import platform
import re
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

from torq.lab import LabError
from torq.lab.pipeline import tools
from torq.lab.pipeline.workflow import ModelPipeline, PipelineConfig

logger = logging.getLogger("torq.lab.debug")

# Shared with torq.testing.issues; torq.lab must not import torq.testing.
ERROR_PATTERN = re.compile(r"(Assertion .*$|error: .*$)", re.MULTILINE)

_TILE_FUSE_SUFFIX = " (encountered while running the pipeline to checking if a tile fits in memory)"

_DIGIT_RUN = re.compile(r"[0-9]+")


def normalize_error(stderr: str) -> Optional[str]:
    """The first Assertion/error line without the tile-and-fuse suffix, digit runs as ``<N>``."""
    match = ERROR_PATTERN.search(stderr)
    if match is None:
        return None
    error = match.group(0).removesuffix(_TILE_FUSE_SUFFIX)
    return _DIGIT_RUN.sub("<N>", error)


@dataclass
class Diagnosis:
    """Compile diagnosis and next step; cls="ok" indicates success."""

    cls: str
    stage: Optional[str] = None
    pass_name: Optional[str] = None
    op: Optional[str] = None
    source_loc: Optional[str] = None
    """Diagnostic file:line:column, or the assertion's file:line."""
    message: Optional[str] = None
    """First Assertion/error line, normalized for deduplication."""
    raw_message: Optional[str] = None
    """First Assertion/error line verbatim (normalization mangles ``bf16``)."""
    op_source: Optional[str] = None
    """Imported ONNX/TFLite op at source_loc, e.g. ``onnx.LpPool``."""
    excerpt: str = ""
    hint: str = ""
    doc: str = ""
    probes: Dict[str, bool] = field(default_factory=dict)
    """Probe flag mapped to whether the compile passed."""
    repro: Tuple[str, ...] = ()
    op_summary: Optional[Dict] = None
    bisect: Optional[List[Dict]] = None
    metadata: Dict = field(default_factory=dict)
    raw_stderr: Optional[str] = None
    """Path of the stage-1 (verdict run) stderr in the bundle."""

    def to_dict(self) -> dict:
        return asdict(self)


class Rule(NamedTuple):
    """A diagnosis class; its doc anchor is the class name."""

    cls: str
    stage: Optional[str]
    pattern: Optional[re.Pattern]
    hint: str


# First match wins. Assertions precede pipeline errors: both can appear, even
# with return code 1. Specific op errors precede legalization and
# serialization wrapper errors.
RULES = (
    Rule("environment", None,
         re.compile(r"Unable to find css config|not found. Set .* TORQ_TOOL_PATH|tosa-converter-for-tflite not found"),
         "Install the missing tool, or fix TORQ_TOOL_PATH / the css config name in --torq-hw."),
    Rule("assert", None,
         re.compile(r"Assertion .* failed"),
         "Compiler bug: quote the assert file:line from the message and report with the crash backtrace."),
    Rule("fatal-error", None,
         re.compile(r"LLVM ERROR:|UNREACHABLE executed"),
         "Compiler bug: report with the stack dump."),
    Rule("invalid-model", "Model validation",
         re.compile(r"ONNX model validation failed"),
         "Fix or re-export the model; onnx.checker names the bad node in the output."),
    Rule("import-failed", "ONNX/TFLite import",
         re.compile(r"import_onnx failed|failed to import ONNX|tosa-converter-for-tflite failed|failed to import TFLite"),
         "Check the opset; the importer's stderr is quoted in the output."),
    Rule("dynamic-shape", "Static-shape gate",
         re.compile(r"does not support dynamic shapes|unhandled dynamic dimensions"),
         "Re-export the model with static shapes and try again."),
    Rule("invalid-source", "input-parse",
         re.compile(r"Unable to parse module assembly|custom op '[a-z0-9_.]+' is unknown"),
         "The input MLIR does not parse with this compiler build; regenerate the MLIR from the model or fix the stale test data."),
    Rule("frontend-unsupported-op", "Frontend lowering",
         re.compile(r"Unimplemented operator"
                    r"|found an op that was marked as backend illegal"
                    r"|failed to legalize operation 'torch\.operator' that was explicitly marked illegal"),
         "The op is missing from the torch-mlir ONNX lowering; re-export around it or request the op."),
    Rule("kernel-unsupported", "Torq codegen",
         re.compile(r"failed to legalize operation 'torq_hl\."),
         "A matcher accepted an op the TorqHW kernel cannot run; check the constraint against https://synaptics-torq.github.io/torq-compiler/v/latest/user-manual/ops.html."),
    Rule("no-npu-lowering", "Torq codegen",
         re.compile(r"'linalg\.[a-z_]+' op unsupported in function body"
                    r"|Unable to find executor for op"
                    r"|Cannot execute operation on (CSS|Host) as disabled by command line option"
                    r"|unable to (tile operation|find memory requirements for op) and cannot fall back to host"),
         "The op has no NPU lowering and its fallback is disabled; re-enable CSS/Host or add an executor-map entry (doc §no-npu-lowering)."),
    Rule("serialization", "Serialization",
         re.compile(r"failed to serialize to torq binary|CSS binary size .* does not match"),
         "Codegen produced IR the backend cannot serialize; an op error (if any) precedes this wrapper line."),
    Rule("memory", "Torq codegen",
         re.compile(r"failed to allocate LRAM addresses"
                    r"|XRAM size exceeded"
                    r"|Failed to allocate (DTCM|ITCM)"
                    r"|Not enough space available for NSS programs"),
         "Memory pressure: try --torq-max-nss-programs-size."),
    Rule("tiling", "Torq codegen",
         re.compile(r"failed to tile reduction|failed to do slicing|failed to slice op to size|failed to peel"),
         "Probe with --torq-disable-slicing."),
    Rule("input-conversion", "Input conversion",
         re.compile(r"conversion failed"
                    r"|cannot preserve I/O dtype"
                    r"|Failed to form single dispatch regions"
                    r"|expected floating point type"),
         "The message names the dtype or op the input-stage conversion cannot lower."),
    Rule("fallback-failed", "Torq codegen",
         re.compile(r"CSS program allocations .* exceed|failed to compile |linking command failed"),
         "A CSS/Host fallback is enabled but cannot build the op; the LLVM error follows."),
    Rule("legalization", "Torq codegen",
         re.compile(r"failed to legalize operation '[a-z0-9_.]+'"),
         "Names the op and the conversion pass; inspect the failed-pass IR."),
    Rule("non-convergence", "Torq codegen",
         re.compile(r"did not converge"),
         "Two rewrite patterns undo each other; use --debug-only=greedy-rewriter or bisect the pattern set."),
)

# Classes decided by return code or by the absence of a matching rule.
_TIMEOUT = Rule("timeout", None, None,
                "The tool hung or a rewrite blew up; the pass above is the last one "
                "started before the timeout. Re-run with a shorter --timeout to confirm.")
_SILENT_PASS_FAILURE = Rule("silent-pass-failure", "Torq codegen", None,
                            "A pass failed without a diagnostic (signalPassFailure); the pass name "
                            "below is the only evidence. Get its IR with --mlir-print-ir-after-failure.")
_CRASH = Rule("crash", None, None,
              "Compiler bug with no diagnostic; re-run with "
              "--mlir-pass-pipeline-crash-reproducer and --torq-show-pass-progress to "
              "name the failing pass, then file an issue.")
_UNKNOWN = Rule("unknown", None, None,
                "No known signature; work the ladder from the excerpt: which pass, which "
                "op, then minimize. If it is a compiler bug, file an issue with the repro.")

# Cross-check the pass chain against the failed IR dump; for crashes,
# use the innermost pass named by the reproducer.
_REPRO_NOTE_SPAN_RE = re.compile(r"Pipeline failed while executing \[(.+?)\]: reproducer generated")
_REPRO_NOTE_NAME_RE = re.compile(r"`([A-Za-z_][\w:]*)` on")

# First ``IR Dump After <Pass> Failed (<name>)`` line; the first occurrence
# is the innermost failing pass.
_IR_DUMP_RE = re.compile(r"IR Dump After \S+ Failed \(([\w-]+)\)")

# ``[TORQ <dispatch>] + <PassName>`` / ``- <PassName> (<ms> ms, <mem>)``.
_PROGRESS_RE = re.compile(r"^\[TORQ [^\]]*\] ([+-]) (\S+)", re.MULTILINE)

_LOC_RE = re.compile(r"([\w./<>-]+):(\d+):(\d+): (?:error|note):")

# Assert lines carry the assert's own source location instead: ``File.cpp:123``.
_ASSERT_LOC_RE = re.compile(r"([\w./<>-]+\.(?:cpp|cc|h|inc|c)):(\d+)")

_OP_NAME_RE = re.compile(r"(?:failed to legalize operation|custom op) '([a-z0-9_.]+)'|'([a-z0-9_.]+)' op ")

# A lone translation wrapper diagnostic indicates silent signalPassFailure.
_WRAPPER_RE = re.compile(r"failed to run translation of source executable to target executable")


def _first_error_line(text: str) -> Optional[str]:
    match = ERROR_PATTERN.search(text)
    if match is None:
        return None
    start = text.rfind("\n", 0, match.start()) + 1
    return text[start:match.end()]


def _extract_source_loc(text: str) -> Optional[str]:
    line = _first_error_line(text)
    if line is None:
        return None
    m = _LOC_RE.search(line)
    if m:
        return f"{m.group(1)}:{m.group(2)}:{m.group(3)}"
    m = _ASSERT_LOC_RE.search(line)
    if m:
        return f"{m.group(1)}:{m.group(2)}"
    return None


def _extract_op(text: str) -> Optional[str]:
    line = _first_error_line(text)
    if line is None:
        return None
    m = _OP_NAME_RE.search(line)
    if m is None:
        return None
    return m.group(1) or m.group(2)


def _progress_pass_name(text: str) -> Optional[str]:
    """Last ``+ <Pass>`` without a matching ``- <Pass>`` in the progress log."""
    open_passes: Dict[str, int] = {}
    for m in _PROGRESS_RE.finditer(text):
        sign, name = m.group(1), m.group(2)
        if sign == "+":
            open_passes[name] = m.start()
        else:
            open_passes.pop(name, None)
    if not open_passes:
        return None
    return max(open_passes, key=open_passes.get)


def _extract_pass_name(text: str) -> Optional[str]:
    dump_m = _IR_DUMP_RE.search(text)
    dump_name = dump_m.group(1) if dump_m else None
    note_m = _REPRO_NOTE_SPAN_RE.search(text)
    if note_m:
        names = _REPRO_NOTE_NAME_RE.findall(note_m.group(1))
        if names:
            if dump_name in names:
                return dump_name
            return names[-1]
    if dump_name:
        return dump_name
    return _progress_pass_name(text)


def _is_silent_pass_failure(text: str) -> bool:
    lines = [line for line in text.splitlines() if "error:" in line]
    return bool(lines) and all(_WRAPPER_RE.search(line) for line in lines)


# Setup progress does not identify a failing pass for pre-pipeline errors.
_PRE_PIPELINE_CLASSES = frozenset(
    {"environment", "invalid-model", "import-failed", "dynamic-shape", "invalid-source"}
)


def _excerpt_line(text: str, pattern: Optional[re.Pattern]) -> str:
    """The first line matching pattern, else the first non-empty line."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return next((line for line in lines if pattern and pattern.search(line)), lines[0] if lines else "")


def _diagnosis_from(rule: Rule, text: str) -> Diagnosis:
    message = normalize_error(text)
    if rule.cls == "unknown":
        tail = "\n".join(text.splitlines()[-10:])
        excerpt = f"{message}\n{tail}" if message else tail
    elif message is not None:
        excerpt = message
    elif rule.cls in ("invalid-model", "import-failed"):
        # LabError text without an error line: the checker/importer message
        # is the actionable content.
        excerpt = "\n".join(text.strip().splitlines()[-30:])
    else:
        excerpt = _excerpt_line(text, rule.pattern)
    return Diagnosis(
        cls=rule.cls,
        stage=rule.stage,
        pass_name=None if rule.cls in _PRE_PIPELINE_CLASSES else _extract_pass_name(text),
        op=_extract_op(text),
        source_loc=_extract_source_loc(text),
        message=message,
        raw_message=_first_error_line(text),
        excerpt=excerpt,
        hint=rule.hint,
        doc=f"https://synaptics-torq.github.io/torq-compiler/v/latest/user-manual/debugging_compile_errors.html#{rule.cls}",
    )


# Known-unsupported configurations that assert on purpose: the class stays
# assert, only the hint changes.
KNOWN_ASSERTS = (
    (re.compile(r"iType == DType::bf16 && wType == DType::bf16"),
     "f32 conv requires bf16 input/weights (convert the model dtype, e.g. torq-lab convert_dtype, and recompile)"),
)


def classify(text: str, returncode: Optional[int]) -> Diagnosis:
    """Classify tool stderr or a LabError message (returncode 1) without running tools.

    A None returncode (timeout) takes precedence over the text. A negative
    returncode without Assertion/error text is a crash; unrecognized
    diagnostics stay unknown even with a signal.
    """
    if returncode is None:
        return _diagnosis_from(_TIMEOUT, text)
    for rule in RULES:
        if rule.pattern.search(text):
            diagnosis = _diagnosis_from(rule, text)
            if rule.cls == "assert":
                raw = _first_error_line(text) or ""
                for known_rx, known_text in KNOWN_ASSERTS:
                    if known_rx.search(raw):
                        diagnosis.hint = f"Known-unsupported configuration, not a compiler bug: {known_text}"
                        break
            return diagnosis
    if returncode == 1 and _is_silent_pass_failure(text):
        return _diagnosis_from(_SILENT_PASS_FAILURE, text)
    if returncode < 0 and not ERROR_PATTERN.search(text):
        return _diagnosis_from(_CRASH, text)
    return _diagnosis_from(_UNKNOWN, text)


# Display-only flags for the stage-1 verdict run: pass progress keeps the
# timeout report nameable, the elision flag keeps console IR short.
_DISPLAY_FLAGS = ("--torq-show-pass-progress", "--mlir-elide-elementsattrs-if-larger=16")

# Bundle flags for the stage-2 run: single-threaded so the failed-pass IR and
# crash reproducer are deterministic. No elision flag: it would corrupt the
# crash reproducer's constants (dense_resource<__elided__>).
_BUNDLE_FLAGS = ("--mlir-disable-threading", "--mlir-print-ir-after-failure", "--mlir-print-debuginfo")


def _add_flags(base: List[str], flags: Tuple[str, ...]) -> List[str]:
    """Append the flags the user did not already pass (in any = form)."""
    opts = list(base)
    for flag in flags:
        if "=" in flag:
            name = flag.split("=", 1)[0]
            if any(o == name or o.startswith(name + "=") for o in opts):
                continue
        elif flag in opts:
            continue
        opts.append(flag)
    return opts


def _strip_elision(base: List[str]) -> List[str]:
    """Drop the user's elision flag: it corrupts a crash reproducer's constants."""
    return [o for o in base if not o.startswith("--mlir-elide-elementsattrs-if-larger=")]


# Probe one flag at a time using the user's options, without diagnostic flags
# (--mlir-print-ir-after-failure suppresses other IR dumps in this build).
# no-npu-lowering uses its fallback probes instead. Classes not listed get
# _DEFAULT_PROBES, the NSS-path isolation probe.
_DEFAULT_PROBES = ("--torq-disable-slices",)
_NSS_PROBES = ("--torq-disable-slices", "--torq-tile-and-fuse-producers-fuse-mode=only-patterns")
PROBE_MAP = {
    "memory": ("--torq-max-nss-programs-size=16777216",),
    "tiling": ("--torq-disable-slicing",),
    "input-conversion": ("--torq-disable-dispatch-fusion",),
    **dict.fromkeys(("kernel-unsupported", "legalization", "non-convergence", "silent-pass-failure"), _NSS_PROBES),
}

# What a passing probe tells you.
PROBE_CONCLUSIONS = {
    "--torq-disable-slices": "the bug is in the NSS path: matcher, kernel, tiling, or memory",
    "--torq-disable-slicing": "the bug is in torq-linalg-slicing",
    "--torq-disable-dispatch-fusion": "the failure is in dispatch fusion, not the op's own conversion",
    "--torq-tile-and-fuse-producers-fuse-mode=only-patterns": "the bug is in the fusion group, not the op",
    "--torq-max-nss-programs-size=16777216": "the default 8 MB NSS program reservation was the limit; 16 MB fits",
}

# Supported-op sections and commonly unmet constraints.
NEAREST_SUPPORTED = {
    "elementwise": {
        "ops_md_section": "Element-Wise Operators",
        "commonly_unmet": [
            "dtype is not in the pattern's supported column",
            "the body is not a recognized elementwise pattern",
        ],
    },
    "reduction": {
        "ops_md_section": "Other Operators",
        "commonly_unmet": ["dtype", "reduction shape/axis"],
    },
    "softmax": {
        "ops_md_section": "Activation related Operators",
        "commonly_unmet": ["dtype", "rank"],
    },
}

_NO_DEBUG_ONLY_NOTE = "matcher reasons need an assert-enabled build"

# A <file>:<line>:<col>: error|note: diagnostic at the start of a line ends
# an IR dump block.
_DIAG_LINE_RE = re.compile(r"^\S+:\d+:\d+: (?:error|note):")

# linalg.generic attributes / operand lists (failed-pass dumps are
# pretty-printed).
_INDEXING_MAPS_RE = re.compile(r"indexing_maps\s*=\s*\[([^\]]*)\]")
_ITERATOR_TYPES_RE = re.compile(r"iterator_types\s*=\s*\[([^\]]*)\]")
_INS_RE = re.compile(r"ins\(([^)]*)\)")
_OUTS_RE = re.compile(r"outs\(([^)]*)\)")

# Element types, so nested memref/tensor attribute commas never split a type.
# No left boundary: in a shape like 1x64xf32 the type follows the 'x';
# bf16 must precede f16 in the alternation.
_ELEMENT_TYPE_RE = re.compile(r"(i\d+|bf16|f16|f32|f64|bool)\b")

# Op-like dotted tokens (pretty-printed bodies); quoted names followed by (
# are the generic-form spelling of the same thing. (?<!#) skips attribute
# namespaces like #torq_hl.executor<css> / #arith.fastmath<none>.
_BODY_OP_RE = re.compile(r"(?<!#)\b([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]+)+)\b")
_QUOTED_OP_RE = re.compile(r'"([a-z][a-z0-9_]*(?:\.[a-z][a-z0-9_]+)+)"(?=\()')

# loc(callsite("file":line:col at ...)): the callsite loc is into the
# work-dir MLIR for ONNX/TFLite sources; the last one on an op is the op's
# own loc.
_CALLSITE_LOC_RE = re.compile(r'loc\(callsite\("([^"]+)":(\d+):(\d+)')
_PLAIN_LOC_RE = re.compile(r'loc\("([^"]+)":(\d+):(\d+)')

_TORCH_OP_RE = re.compile(r'torch\.operator "onnx\.(\w+)"')

# torq_hl.program "<name>": <executor>_<op>_<id> for fallbacks, NSS
# dispatches carry _dispatch_N.
_PROGRAM_RE = re.compile(r'torq_hl\.program "([^"]+)"')

# Generic-form trailing signature: `... } : (operand types) -> result type`.
_GENERIC_SIG_RE = re.compile(r": \(([^)]*)\) -> (\S+)")

# --debug-only pattern trace (MLIR rewriter):
#   * Pattern : 'linalg.generic -> ()' {
#     ** Failure : <notifyMatchFailure reason>
_PATTERN_HEADER_RE = re.compile(r"\* Pattern : '([\w.]+)")
_MATCHER_FAILURE_RE = re.compile(r"\*\* Failure : (.+)")


def _crash_repro_cmd(cmd: List[str], crash_repro: Path) -> str:
    """Build a standalone crash-reproducer command, preserving compiler flags.

    Stateful options such as --torq-executor-map are absent from the
    reproducer and must be passed again. Display flags (pass progress,
    elision) are stripped: the reproducer is a minimal, clean re-fire.
    """
    args = [str(a) for a in cmd]
    args = [a for a in args if not a.startswith(("--torq-show-pass-progress", "--mlir-elide-elementsattrs-if-larger"))]
    replaced_source = False
    for i in range(1, len(args)):  # args[0] is the tool itself
        if args[i] == "-o":
            args[i + 1] = "/tmp/crash-repro.vmfb"
        elif not args[i].startswith("-") and not replaced_source:
            args[i] = str(crash_repro)
            replaced_source = True
    return " ".join(args)


def _extract_failed_pass_ir(stderr: str) -> Optional[str]:
    m = _IR_DUMP_RE.search(stderr)
    if m is None:
        return None
    start = stderr.rfind("\n", 0, m.start()) + 1
    lines = stderr[start:].splitlines()
    out: List[str] = []
    for i, line in enumerate(lines):
        if i > 0 and (
            not line.strip()
            or line.startswith(("[TORQ ", "Command:"))
            or _DIAG_LINE_RE.match(line)
        ):
            break
        out.append(line)
    return "\n".join(out) + "\n" if out else None


def _extract_current_op_section(stderr: str) -> Optional[str]:
    """Return the first "see current operation" section when no IR dump exists."""
    m = re.search(r"note: see current operation:\s*\n", stderr)
    if m is None:
        return None
    rest = stderr[m.end():]
    end = re.search(r"(?m)^\S+:\d+:\d+: (?:error|note|warning):", rest)
    return rest[: end.start()] if end else rest


def _find_failing_op(ir: str, op_name: str) -> Optional[str]:
    """Find the first statement for op_name, including its region if present."""
    name = re.escape(op_name)
    rx = re.compile(
        rf'(?m)^\s*(?:%\w+\s*=\s*)?"{name}"(?=\()'
        rf"|^\s*(?:%\w+\s*=\s*)?(?<![\w.])" + name + r"(?=\s*[\{(])"
    )
    m = rx.search(ir)
    if m is None:
        return None
    rest = ir[m.end():]
    if re.search(r"\n\s*\^", rest):  # block label: this op carries a region
        close = re.search(r"(?m)^\s*\}[^\n]*\n?", rest)
        end = m.end() + (close.end() if close else len(rest))
    else:
        eol = rest.find("\n")
        end = m.end() + (eol if eol != -1 else len(rest))
    return ir[m.start():end]


def _body_ops(body: str) -> List[str]:
    """Region op names minus linalg.yield, in order of first appearance."""
    unquoted = re.sub(r'"[^"\n]*"', " ", body)
    names = [m.group(1) for m in _BODY_OP_RE.finditer(unquoted)] + _QUOTED_OP_RE.findall(body)
    return [n for n in dict.fromkeys(names) if n != "linalg.yield"]


def _split_top(s: str) -> List[str]:
    """Comma-split at paren depth 0 (affine-map commas sit inside parens;
    angles are not counted: ``->`` would unbalance them)."""
    parts, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p.strip().strip('"') for p in parts if p.strip()]


def _element_types(text: Optional[str]) -> List[str]:
    """Element type tokens (f32, i8, ...) in an ins/outs operand list."""
    return list(dict.fromkeys(_ELEMENT_TYPE_RE.findall(text or "")))


def _op_loc(op_text: str) -> Optional[str]:
    """The op's own loc: <file>:<line>:<col>."""
    callsites = _CALLSITE_LOC_RE.findall(op_text)
    if callsites:
        f, ln, col = callsites[-1]
        return f"{f}:{ln}:{col}"
    m = _PLAIN_LOC_RE.search(op_text)
    if m:
        return f"{m.group(1)}:{m.group(2)}:{m.group(3)}"
    return None


def _imported_line(pipe, loc: str) -> Optional[str]:
    """The ``<work-dir>/<model-stem>.mlir`` line loc points into (ONNX/TFLite sources only)."""
    model = Path(pipe.config.model_path)
    if model.suffix not in (".onnx", ".tflite"):
        return None
    try:
        _file, line_s, _col = loc.rsplit(":", 2)
        imported = Path(pipe.work_dir) / f"{model.stem}.mlir"
        return imported.read_text().splitlines()[int(line_s) - 1]
    except (OSError, ValueError, IndexError):
        return None


def _source_op(pipe, loc: str) -> Optional[str]:
    """The ONNX op type at loc; imported MLIR keeps op types but not node names."""
    line = _imported_line(pipe, loc)
    m = _TORCH_OP_RE.search(line) if line else None
    return f"onnx.{m.group(1)}" if m else None


def _source_node(pipe, loc: str) -> str:
    op = _source_op(pipe, loc)
    if op is None:
        return loc
    file, line, col = loc.rsplit(":", 2)
    return f"{op} at {Path(file).name}:{line}:{col}"


def _compile_with(pipe, options: List[str]) -> Tuple[str, Optional[int]]:
    """Compile with options; return diagnostics and return code (1 for a LabError).

    Leaves options set; diagnose() restores the user's options on exit.
    """
    pipe.config.compiler_options = options
    try:
        result = pipe.compile()
        return result.diagnostics, 0
    except tools.ToolError as exc:
        return exc.output, exc.returncode
    except LabError as exc:
        return str(exc), 1


def _probe_options(base: List[str], flag: str) -> Optional[List[str]]:
    """The user's options plus flag (replacing an =-form variant).

    None when the user already passed flag: the re-run would repeat the failed compile.
    """
    if "=" in flag:
        name = flag.split("=", 1)[0]
        opts = [o for o in base if o != name and not o.startswith(name + "=")]
    elif flag in base:
        return None
    else:
        opts = list(base)
    opts.append(flag)
    return opts


def _flag_probes(pipe, cls: str, base_options: List[str]) -> Dict[str, bool]:
    results: Dict[str, bool] = {}
    for flag in PROBE_MAP.get(cls, _DEFAULT_PROBES):
        opts = _probe_options(base_options, flag)
        if opts is not None:
            results[flag] = _compile_with(pipe, opts)[1] == 0
    return results


def _parse_matcher_reasons(output: str, op_name: str) -> List[str]:
    """notifyMatchFailure reasons for op_name in a --debug-only trace."""
    reasons: List[str] = []
    current: Optional[str] = None
    for line in output.splitlines():
        pm = _PATTERN_HEADER_RE.search(line)
        if pm:
            current = pm.group(1)
            continue
        fm = _MATCHER_FAILURE_RE.search(line)
        if fm and current == op_name and fm.group(1).strip() not in reasons:
            reasons.append(fm.group(1).strip())
    return reasons


def _executor_from_ir(output: str, op_name: Optional[str]) -> Optional[str]:
    """The executor of the program named after op_name, else of the first program
    (one-op models may name it after the function)."""
    names = _PROGRAM_RE.findall(output)
    if not names:
        return None

    def _executor_of(name: str) -> Optional[str]:
        if name.startswith("host_"):
            return "host"
        if name.startswith("css_"):
            return "css"
        if "_dispatch_" in name:
            return "nss"
        return None

    for name in names:
        if op_name and op_name in name:
            return _executor_of(name)
    return _executor_of(names[0])


def _op_kind(op_name: str, iterator_types: List[str]) -> str:
    if op_name == "linalg.softmax":
        return "softmax"
    if any("reduction" in t for t in iterator_types):
        return "reduction"
    return "elementwise"


def _fallback_probes(pipe, op_name: Optional[str], options: List[str]) -> Dict[str, Dict[str, object]]:
    """Re-enable each disabled fallback separately; report pass/fail and the executor.

    Drops --mlir-print-ir-after-failure, which suppresses the assign-pass IR.
    """
    results: Dict[str, Dict[str, object]] = {}
    for flag in ("--torq-disable-css", "--torq-disable-host"):
        if flag not in options:
            continue
        opts = [o for o in options if o not in (flag, "--mlir-print-ir-after-failure")]
        output, rc = _compile_with(pipe, opts + ["--mlir-print-ir-after=torq-assign-operations-to-cpu-programs"])
        results[flag] = {
            "passed": rc == 0,
            "executor": _executor_from_ir(output, op_name) if rc == 0 else None,
        }
    return results


def _build_op_summary(pipe, stderr: str, op_name: Optional[str],
                      fallback_loc: Optional[str] = None) -> Optional[Dict]:
    """Summarize the failing op, source, fallback results, and matcher reasons.

    None if neither the failed-pass IR nor the current-op note contains the op.
    """
    options = list(pipe.config.compiler_options)
    ir = _extract_failed_pass_ir(stderr)
    text = ir if ir is not None else _extract_current_op_section(stderr)
    if text is None:
        return None
    if op_name is None:  # the error line named no op: take the first linalg.*
        for m in re.finditer(r"\b(linalg\.[a-z][\w]*)\b", text):
            if m.group(1) not in ("linalg.yield", "linalg.index"):
                op_name = m.group(1)
                break
    if op_name is None:
        return None
    if ir is not None:
        pipe.debug_dir.mkdir(parents=True, exist_ok=True)
        (pipe.debug_dir / "failed-pass.mlir").write_text(ir)
    op_text = _find_failing_op(text, op_name)
    if op_text is None:
        return None

    maps_m = _INDEXING_MAPS_RE.search(op_text)
    iters_m = _ITERATOR_TYPES_RE.search(op_text)
    maps = _split_top(maps_m.group(1)) if maps_m else []
    iters = _split_top(iters_m.group(1)) if iters_m else []
    body = ""
    bb = re.search(r"\n\s*\^", op_text)
    if bb:
        body = op_text[bb.end():]
    ins_m = _INS_RE.search(op_text)
    outs_m = _OUTS_RE.search(op_text)
    if ins_m is not None:
        operand_types = _element_types(ins_m.group(1))
        result_types = _element_types(outs_m.group(1) if outs_m else None)
    else:
        sig_m = _GENERIC_SIG_RE.search(op_text)
        if sig_m:
            operand_types = _element_types(sig_m.group(1))
            result_types = _element_types(sig_m.group(2))
        else:
            operand_types, result_types = [], []
    loc = _op_loc(op_text) or fallback_loc

    summary: Dict = {
        "op": op_name,
        "indexing_maps": maps,
        "iterator_types": iters,
        "operand_element_types": operand_types,
        "result_element_types": result_types,
        "body_ops": _body_ops(body),
        "loc": loc,
        "source": _source_node(pipe, loc) if loc else None,
        "nearest_supported": NEAREST_SUPPORTED[_op_kind(op_name, iters)],
    }

    fallback = _fallback_probes(pipe, op_name, options)
    summary["fallback"] = fallback
    summary["probes"] = {flag: bool(r["passed"]) for flag, r in fallback.items()}

    # Matcher traces require an assertion-enabled compiler.
    debug_only = bool(pipe.debug_only_available())
    if debug_only:
        output, _rc = _compile_with(pipe, options + ["--debug-only=linalg-torq-*,dialect-conversion"])
        summary["matcher_reasons"] = _parse_matcher_reasons(output, op_name)
    else:
        summary["matcher_reasons"] = [_NO_DEBUG_ONLY_NOTE]
    summary["debug_only_available"] = debug_only
    return summary


def _bisect_layer(pipe, bisect_dir: Path, layer_name: str, layer, onnx,
                  base_options: List[str]) -> Dict:
    layer_dir = bisect_dir / layer_name
    layer_dir.mkdir(parents=True, exist_ok=True)
    layer_model = layer.model if hasattr(layer, "model") else layer
    onnx_path = layer_dir / f"{layer_name}.onnx"
    onnx.save(layer_model, str(onnx_path))
    layer_config = PipelineConfig(
        model_path=onnx_path,
        work_dir=layer_dir,
        chip=pipe.config.chip,
        runtime_hw_type=pipe.config.runtime_hw_type,
        compiler_options=[*_strip_elision(base_options), *_BUNDLE_FLAGS,
                          f"--mlir-pass-pipeline-crash-reproducer={layer_dir / 'debug' / 'crash-repro.mlir'}"],
        timeout=pipe.config.timeout,
    )
    layer_pipe = ModelPipeline(layer_config)
    try:
        layer_pipe.compile()
        return {"layer": layer_name, "cls": "ok", "message": None}
    except tools.ToolError as exc:
        diagnosis = classify(exc.output, exc.returncode)
    except Exception as exc:  # gate/import/parse failures, like diagnose()
        diagnosis = classify(str(exc), 1)
    return {"layer": layer_name, "cls": diagnosis.cls, "message": diagnosis.message}


def _bisect_layers(pipe, base_options: List[str]) -> List[Dict]:
    """Compile every extracted ONNX layer and return its diagnosis."""
    import onnx
    from torq.lab.model_tools.extraction.onnx import layers as onnx_layers

    full_model = onnx_layers.get_full_model(str(pipe.config.model_path))
    layers = onnx_layers.generate_onnx_layers_from_model(full_model)
    bisect_dir = pipe.work_dir / "bisect"
    return [_bisect_layer(pipe, bisect_dir, name, layer, onnx, base_options)
            for name, layer in layers.items()]


def _tool_version(tool: str) -> Optional[str]:
    """The compiler's ``--version`` output block, or None on any failure."""
    try:
        proc = subprocess.run([tool, "--version"], capture_output=True, text=True, timeout=30)
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def _validate_crash_repro(path: Path) -> str:
    """Bundle-metadata status of a crash reproducer: no elided content and it parses."""
    try:
        text = path.read_text(errors="replace")
    except OSError:
        return "unavailable"
    if "__elided__" in text:
        return "failed: elided content detected"
    try:
        from iree.compiler import ir
    except ImportError:
        return "unavailable"
    try:
        with ir.Context() as ctx:
            ir.Module.parse(text, context=ctx)
    except Exception as exc:
        return "failed: " + str(exc).splitlines()[0]
    return "ok"


def _write_bundle(pipe, diagnosis: Diagnosis, cmd: Optional[List[str]], rc: Optional[int]) -> Tuple[str, ...]:
    """Write diagnosis.json and repro.sh, and return the bundle's artifact paths.

    ``cmd`` is the stage-1 command, i.e. the user's failing compile; the crash
    reproducer and failed-pass IR come from the stage-2 run.
    """
    debug_dir = pipe.debug_dir
    debug_dir.mkdir(parents=True, exist_ok=True)
    paths = [str(debug_dir / "diagnosis.json")]
    if cmd is not None:
        paths.append(str(debug_dir / "repro.sh"))
    raw_stderr = debug_dir / "raw-stderr.txt"
    if raw_stderr.exists():
        paths.append(str(raw_stderr))
    failed_pass = debug_dir / "failed-pass.mlir"
    if failed_pass.exists():
        paths.append(str(failed_pass))
    crash_repro = debug_dir / "crash-repro.mlir"
    if crash_repro.exists():
        paths.append(str(crash_repro))
        diagnosis.metadata["crash_repro_parse"] = _validate_crash_repro(crash_repro)
    diagnosis.repro = tuple(paths)
    metadata = diagnosis.metadata
    metadata["tool"] = str(cmd[0]) if cmd is not None else None
    metadata["tool_version"] = _tool_version(str(cmd[0])) if cmd is not None else None
    metadata.update(
        chip=pipe.config.chip,
        runtime_hw_type=pipe.config.runtime_hw_type,
        platform=platform.platform(),
        machine=platform.machine(),
        created=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    (debug_dir / "diagnosis.json").write_text(json.dumps(diagnosis.to_dict(), indent=2) + "\n")
    if cmd is not None:
        if rc is None:
            expected = "expected to time out"  # the verdict run hit --timeout
        elif rc < 0:
            expected = f"expected to crash (shell exit code {128 - rc})"  # signal
        else:
            expected = f"expected exit code {rc}"
        lines = [
            "#!/bin/sh",
            f"# torq-lab debug repro: {diagnosis.cls}",
            "",
            f"# 1. the user's failing compile ({expected})",
            " ".join(str(c) for c in cmd),
            "",
        ]
        if crash_repro.exists():
            lines.append("# 2. standalone crash reproducer (expected to re-fire the same failure; stateful flags re-passed):")
            lines.append(_crash_repro_cmd(cmd, crash_repro))
            lines.append("")
        repro_sh = debug_dir / "repro.sh"
        repro_sh.write_text("\n".join(lines))
        repro_sh.chmod(0o755)
    return tuple(paths)


def diagnose(pipe, probe: bool = False, bisect: bool = False) -> Diagnosis:
    """Compile in two stages, classify the failure, and write a repro bundle.

    Stage 1 (verdict) compiles with the user's options plus display flags and
    alone decides the class, so instrumentation cannot flip it. Stage 2
    (bundle) recompiles single-threaded with IR dumps and the crash
    reproducer; it is skipped for memory (single-threaded runs hang there)
    and after a timeout. no-npu-lowering always gets fallback probes; probe
    and bisect add flag probes and per-layer ONNX compiles. The user's
    compiler options are restored on return.
    """
    base_options = list(pipe.config.compiler_options)
    try:
        return _diagnose(pipe, base_options, probe, bisect)
    finally:
        pipe.config.compiler_options = base_options


def _diagnose(pipe, base_options: List[str], probe: bool, bisect: bool) -> Diagnosis:
    pipe._ensure_work_dir()
    pipe.debug_dir.mkdir(parents=True, exist_ok=True)
    # A rerun (e.g. --reuse-work-dir) must not report a stale bundle.
    for stale in ("diagnosis.json", "failed-pass.mlir", "repro.sh", "crash-repro.mlir", "raw-stderr.txt"):
        (pipe.debug_dir / stale).unlink(missing_ok=True)

    pipe.config.compiler_options = _add_flags(base_options, _DISPLAY_FLAGS)
    try:
        result = pipe.compile()
    except tools.ToolError as exc:
        output, rc, cmd = exc.output, exc.returncode, exc.cmd
    except LabError as exc:  # import failure or the static-shape gate inside compile()
        output, rc, cmd = str(exc), 1, None
    except Exception as exc:
        # MLIR bindings can raise parse errors outside the LabError hierarchy.
        logger.debug("diagnose: non-tool exception from compile: %r", exc)
        output, rc, cmd = str(exc), 1, None
    else:
        return Diagnosis(cls="ok", hint=f"Compiled clean; run it: torq-lab run {result.vmfb_path}")

    diagnosis = classify(output, rc)
    raw_stderr = pipe.debug_dir / "raw-stderr.txt"
    raw_stderr.write_text(output)
    diagnosis.raw_stderr = str(raw_stderr)

    # Stage 2 is best effort: whatever artifacts it produced are used.
    stage2_output = None
    if cmd is not None and rc is not None and diagnosis.cls != "memory":
        crash_repro_flag = f"--mlir-pass-pipeline-crash-reproducer={pipe.debug_dir / 'crash-repro.mlir'}"
        pipe.config.compiler_options = (
            _add_flags(_strip_elision(base_options), _BUNDLE_FLAGS) + [crash_repro_flag]
        )
        try:
            result = pipe.compile()
            stage2_output = result.diagnostics
        except tools.ToolError as exc:
            stage2_output = exc.output
        except Exception as exc:
            logger.debug("diagnose: non-tool exception from bundle compile: %r", exc)

    # The op summary wants the non-elided failed-pass IR that only stage 2 dumps.
    op_text = stage2_output if stage2_output is not None else output
    if diagnosis.pass_name is None and stage2_output is not None:
        # IREE passes print no TORQ progress lines; stage 2's IR dump or
        # reproducer note names them.
        diagnosis.pass_name = _extract_pass_name(stage2_output)
    if diagnosis.cls == "no-npu-lowering":
        diagnosis.op_summary = _build_op_summary(pipe, op_text, diagnosis.op,
                                                 fallback_loc=diagnosis.source_loc)
        if diagnosis.op_summary:
            diagnosis.probes = dict(diagnosis.op_summary.pop("probes"))
    elif probe and cmd is not None and rc is not None:
        diagnosis.probes = _flag_probes(pipe, diagnosis.cls, base_options)
    if diagnosis.op == "torch.operator" and diagnosis.source_loc:
        diagnosis.op_source = _source_op(pipe, diagnosis.source_loc)
    if bisect and Path(pipe.config.model_path).suffix == ".onnx":
        try:
            diagnosis.bisect = _bisect_layers(pipe, base_options)
        except Exception as exc:  # diagnose() raises nothing
            logger.warning("bisect: layer extraction failed: %s", exc)
    _write_bundle(pipe, diagnosis, cmd, rc)
    return diagnosis
