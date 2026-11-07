#!/usr/bin/env python3
"""Rewrite TOSA ops from the IREE 3.12 syntax (properties in an attribute dictionary) to the
IREE 3.13 syntax (properties after the operands, LLVM b9da7a34df):

  tosa.conv2d %a, %b, ... {acc_type = i32, dilation = array<i64: 1, 1>, ...} : ...
  ->
  tosa.conv2d %a, %b, ... pad([...]) stride([...]) dilation([1, 1]) acc_type(i32) : ...

It also rewrites the elementwise linalg named ops that IREE 3.13 removed (linalg.reciprocal,
linalg.add, linalg.max, ...) to linalg.elementwise:

  linalg.reciprocal ins(%x : T) outs(%y : T) -> T
  ->
  linalg.elementwise <reciprocal> ins(%x : T) outs(%y : T) -> T

Files are rewritten in place. Comments, layout and CRLF line endings are kept, and only TOSA
ops written with an attribute dictionary change. Generic forms ("tosa.x"(...) <{...}>) still
parse and are left alone.

Ops in a TOSA spec older than the one IREE 3.12 accepts (e.g. tosa.fully_connected,
quantization_info, double_round) did not parse with IREE 3.12 either. They are left alone and
reported: regenerate those files from their source model.

After rewriting, each file is parsed with iree-opt, and the files that still don't parse are
reported. The script uses the first iree-opt of IREE 3.13 or later it finds in:
  1. the build directory registered in the Python venv by scripts/configure_python.sh
  2. $IREE_BUILD_DIR
  3. ../iree-build, next to this checkout
  4. PATH
The check is skipped, with a note, when none is found.

Usage:
  migrate_tosa_syntax.py [--iree-opt PATH] [--no-check] [--dry-run] FILE...

  --iree-opt PATH  iree-opt to check with, instead of searching for one
                   (e.g. <build-dir>/third_party/iree/tools/iree-opt)
  --no-check       don't parse the rewritten files
  --dry-run        report what would change, don't write anything (no check)
"""
import argparse
import glob
import os
import re
import shutil
import site
import subprocess
import sys

# op -> its properties, in the order of the IREE 3.13 assembly format. Styles:
#   "paren": key(value)    "angle": key<value> (enums)
#   "opt":   key(value), optional; omitted when absent
P, A, O = "paren", "angle", "opt"
CONV = [("pad", P), ("stride", P), ("dilation", P), ("acc_type", P), ("local_bound", O)]
AXIS = [("axis", P)]
AXIS_NAN = [("axis", P), ("nan_mode", A)]
NAN = [("nan_mode", A)]
BLOCK = [("block_size", A)]
FORMATS = {
    "argmax": AXIS_NAN,
    "argmin": AXIS_NAN,
    "avg_pool2d": [("kernel", P), ("stride", P), ("pad", P), ("acc_type", P)],
    "avg_pool2d_adaptive": [("acc_type", P)],
    "conv2d": CONV,
    "conv2d_block_scaled": BLOCK,
    "conv3d": CONV,
    "depthwise_conv2d": CONV,
    "fft2d": [("inverse", P), ("local_bound", O)],
    "matmul_t_block_scaled": BLOCK,
    "max_pool2d": [("kernel", P), ("stride", P), ("pad", P), ("nan_mode", A)],
    "max_pool2d_adaptive": NAN,
    "rfft2d": [("local_bound", O)],
    "transpose_conv2d": [("out_pad", P), ("stride", P), ("acc_type", P), ("local_bound", O)],
    "clamp": [("min_val", P), ("max_val", P), ("nan_mode", A)],
    "arithmetic_right_shift": [("round", P)],
    "maximum": NAN,
    "minimum": NAN,
    "reduce_all": AXIS,
    "reduce_any": AXIS,
    "reduce_max": AXIS_NAN,
    "reduce_min": AXIS_NAN,
    "reduce_product": AXIS,
    "reduce_sum": AXIS,
    "concat": AXIS,
    "reshape_block_scaled": BLOCK,
    "reverse": AXIS,
    "transpose": [("perms", P)],
    "row_gather_block_scaled": BLOCK,
    "resize": [("mode", A)],
    "cast": [("input_unsigned", O)],
    "cast_from_block_scaled": BLOCK,
    "cast_to_block_scaled": BLOCK,
    "rescale": [("scale32", P), ("rounding_mode", A), ("per_channel", P),
                ("input_unsigned", P), ("output_unsigned", P)],
    "const": [("values", P)],
    "custom": [("operator_name", P), ("domain_name", P), ("implementation_attrs", P)],
    "apply_scale": [("rounding_mode", A)],
    "const_shape": [("values", P)],
    "dim": AXIS,
    "assert_equal_shape": [("allow_broadcast", P)],
}
# Enum properties with a default value: like "opt" ones, they may be missing from the old
# dictionary.
OPTIONAL_ENUMS = {"nan_mode"}
# Ops whose IREE 3.13 format still has an attribute dictionary in the same place.
NO_PROPERTIES_OK = {"variable", "variable_write", "variable_read", "yield", "cond_if",
                    "while_loop"}
# Ops and attributes of TOSA specs older than the one IREE 3.12 accepts. The attributes only
# explain a report: on ops that have all their properties they are discardable and kept.
LEGACY_OPS = {"fully_connected"}
LEGACY_ATTRS = {"quantization_info", "double_round", "multiplier", "shift", "input_zp",
                "output_zp", "weight_zp", "min_int", "max_int", "min_fp", "max_fp",
                "new_shape", "multiples", "start", "size", "padding", "pad_const",
                "out_shape", "scale", "offset", "border", "value"}

# Elementwise linalg named ops removed in IREE 3.13 (LLVM 80164074c0) -> linalg.elementwise kind.
LINALG_KINDS = {op: op for op in [
    "exp", "log", "abs", "ceil", "floor", "negf", "reciprocal", "round", "sqrt", "rsqrt",
    "square", "tanh", "erf", "add", "sub", "mul", "div", "div_unsigned", "powf", "select"]}
LINALG_KINDS.update({"max": "max_signed", "min": "min_signed"})

OP_RE = re.compile(r'(?<!["\w.#])tosa\.(\w+)\b')
LINALG_RE = re.compile(r'(?<!["\w.#])linalg\.(' + "|".join(LINALG_KINDS) + r')(?=\s+ins\s*\()')
OPEN = {"(": ")", "[": "]", "{": "}", "<": ">"}


def match_close(text, i):
    """Index just past the bracket group starting at text[i]."""
    stack = [OPEN[text[i]]]
    j = i + 1
    while j < len(text) and stack:
        c = text[j]
        if c == '"':
            j = text.index('"', j + 1)
        elif c == ">" and stack[-1] != ">" and text[j - 1] == "-":
            pass  # '->' inside a type
        elif c in OPEN:
            stack.append(OPEN[c])
        elif c == stack[-1]:
            stack.pop()
        j += 1
    return j


def split_top(s, sep=","):
    parts, depth, cur, k = [], 0, "", 0
    while k < len(s):
        c = s[k]
        if c == '"':
            e = s.index('"', k + 1)
            cur += s[k:e + 1]
            k = e + 1
            continue
        if c in "([{<":
            depth += 1
        elif c in ")]}>" and not (c == ">" and k and s[k - 1] == "-"):
            depth -= 1
        if c == sep and depth == 0:
            parts.append(cur)
            cur = ""
        else:
            cur += c
        k += 1
    if cur.strip():
        parts.append(cur)
    return parts


def convert_value(key, style, value):
    value = value.strip()
    m = re.fullmatch(r"array<i(?:32|64)(?::\s*(.*))?>", value, re.S)
    if m:
        return "[" + (m.group(1) or "").strip() + "]"
    if key == "axis":
        return re.sub(r"\s*:\s*i32$", "", value)
    if style == A:
        # Enum values may be spelled in full (#tosa.rounding_mode<DOUBLE_ROUND>) or quoted.
        m = re.fullmatch(r'#tosa\.\w+<(\w+)>|"(\w+)"', value)
        if m:
            return m.group(1) or m.group(2)
    return value


def convert_dict(op, body):
    """Return (new_text, problem) for the attribute dictionary body of a TOSA op."""
    attrs = {}
    for part in split_top(body):
        key, _, value = part.partition("=")
        attrs[key.strip()] = value.strip()
    missing = [k for k, style in FORMATS[op]
               if style != O and k not in OPTIONAL_ENUMS and k not in attrs]
    if missing:
        legacy = sorted(LEGACY_ATTRS & attrs.keys())
        what = f"lacks {', '.join(missing)}" + (f", has {', '.join(legacy)}" if legacy else "")
        return None, f"{what}: older TOSA spec, didn't parse with IREE 3.12 either"
    out = []
    for key, style in FORMATS[op]:
        if key not in attrs:
            continue
        v = convert_value(key, style, attrs.pop(key))
        out.append(f"{key}<{v}>" if style == A else f"{key}({v})")
    if attrs:  # discardable attributes stay in the attribute dictionary
        out.append("{" + ", ".join(f"{k} = {v}" for k, v in attrs.items()) + "}")
    return " ".join(out), None


def migrate(text, report):
    pos, out = 0, []
    for m in OP_RE.finditer(text):
        if m.start() < pos:
            continue
        op = m.group(1)
        line = text.count("\n", 0, m.start()) + 1
        if op in LEGACY_OPS:
            report.append(f"  line {line}: tosa.{op} left as is: older TOSA spec, didn't parse "
                          f"with IREE 3.12 either")
            continue
        if op in NO_PROPERTIES_OK:
            continue
        # Scan the operand list up to the attribute dictionary or the ':' of the type.
        j = m.end()
        while j < len(text) and text[j] not in "{:\r\n":
            j += 1
        if j >= len(text) or text[j] != "{":
            continue  # no attribute dictionary (new syntax already, or no properties)
        end = match_close(text, j)
        if not text[end:].lstrip(" \r\n").startswith(":"):
            continue  # a region, not an attribute dictionary
        if op not in FORMATS:
            continue  # no properties: the attribute dictionary still parses
        new, problem = convert_dict(op, text[j + 1:end - 1])
        if new is None:
            report.append(f"  line {line}: tosa.{op} left as is: {problem}")
            continue
        operands = text[m.end():j].rstrip()
        out.append(text[pos:m.start()])
        out.append(f"tosa.{op}{operands} {new} ")
        pos = end
        while pos < len(text) and text[pos] == " ":
            pos += 1
    out.append(text[pos:])
    text = "".join(out)
    return LINALG_RE.sub(lambda m: f"linalg.elementwise <{LINALG_KINDS[m.group(1)]}>", text)


# Parses only with IREE 3.13 or later: used to tell an older iree-opt apart.
PROBE = """func.func @probe(%a: tensor<2xf32>) -> tensor<2xf32> {
  %0 = tosa.clamp %a min_val(0.0 : f32) max_val(1.0 : f32) : (tensor<2xf32>) -> tensor<2xf32>
  return %0 : tensor<2xf32>
}
"""


def parse_error(iree_opt, path, stdin=None):
    """Return the first error line of iree-opt on the file, or None if it parses."""
    r = subprocess.run([iree_opt, path, "-o", "/dev/null"], input=stdin, capture_output=True,
                       text=True)
    if r.returncode == 0:
        return None
    errors = [l for l in r.stderr.splitlines() if "error:" in l]
    return (errors or r.stderr.splitlines() or ["iree-opt failed"])[0].strip()


TOOL = os.path.join("third_party", "iree", "tools", "iree-opt")
BINDINGS = os.path.join("third_party", "iree", "compiler", "bindings", "python")


def venv_build_dirs():
    """Build directories registered in the venv by scripts/configure_python.sh (iree.pth)."""
    site_dirs = list(site.getsitepackages())
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        site_dirs += glob.glob(os.path.join(venv, "lib", "python3*", "site-packages"))
    dirs = []
    for d in site_dirs:
        try:
            lines = open(os.path.join(d, "iree.pth")).read().splitlines()
        except OSError:
            continue
        dirs += [l.strip()[:-len(BINDINGS)] for l in lines if l.strip().endswith(BINDINGS)]
    return dirs


def iree_opt_candidates():
    """iree-opt binaries to try, most likely to match this checkout first."""
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    build_dirs = venv_build_dirs()
    if os.environ.get("IREE_BUILD_DIR"):
        build_dirs.append(os.environ["IREE_BUILD_DIR"])
    build_dirs.append(os.path.join(os.path.dirname(repo), "iree-build"))
    found = [os.path.join(d, TOOL) for d in build_dirs] + [shutil.which("iree-opt")]
    seen, result = set(), []
    for f in found:
        if f and os.access(f, os.X_OK) and os.path.realpath(f) not in seen:
            seen.add(os.path.realpath(f))
            result.append(f)
    return result


def is_iree_313(iree_opt):
    try:
        return parse_error(iree_opt, "-", stdin=PROBE) is None
    except OSError:
        return False


def find_iree_opt(path):
    """Return the iree-opt to check with, or None after printing why there is none."""
    candidates = [path] if path else iree_opt_candidates()
    for iree_opt in candidates:
        if is_iree_313(iree_opt):
            print(f"checking with {iree_opt}", file=sys.stderr)
            return iree_opt
    hint = "build with IREE 3.13 first, or pass --iree-opt <build-dir>/" + TOOL
    if candidates:
        print(f"note: {', '.join(candidates)}: not runnable or older than IREE 3.13, files not "
              f"checked ({hint})", file=sys.stderr)
    else:
        print(f"note: no iree-opt found, files not checked ({hint})", file=sys.stderr)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--iree-opt", metavar="PATH",
                        help="iree-opt to check with, instead of searching for one")
    parser.add_argument("--no-check", action="store_true", help="don't parse the files")
    parser.add_argument("--dry-run", action="store_true", help="don't write anything")
    parser.add_argument("files", nargs="+", metavar="FILE")
    args = parser.parse_args()
    iree_opt = None if args.no_check or args.dry_run else find_iree_opt(args.iree_opt)

    changed, flagged, failed = 0, 0, 0
    for path in args.files:
        # newline="": keep CRLF files as they are.
        text = open(path, newline="").read()
        report = []
        new = migrate(text, report)
        if new != text:
            changed += 1
            if not args.dry_run:
                open(path, "w", newline="").write(new)
        error = parse_error(iree_opt, path) if iree_opt else None
        if report or error:
            print(path)
            print("\n".join(report + ([f"  doesn't parse: {error}"] if error else [])))
            flagged += bool(report)
            failed += bool(error)
    verb = "would be rewritten" if args.dry_run else "rewritten"
    summary = f"{changed} of {len(args.files)} file(s) {verb}, {flagged} with ops left as is"
    if iree_opt:
        summary += f", {failed} that don't parse"
    print(summary, file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
