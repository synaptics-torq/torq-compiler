# Gen-Config Architecture

---

## Table of Contents

1. [Overview](#1-overview)
2. [Module Map](#2-module-map)
3. [Dependency Graph](#3-dependency-graph)
4. [Two-JSON Design](#4-two-json-design)
5. [Quantization Support](#5-quantization-support)
6. [CLI Subcommand Flow](#6-cli-subcommand-flow)
7. [How to Add a New Model Format](#7-how-to-add-a-new-model-format)
8. [How to Add a New CLI Subcommand](#8-how-to-add-a-new-cli-subcommand)
9. [Testing Architecture](#9-testing-architecture)
10. [Standalone CLI Options](#10-standalone-cli-options)
11. [TFLite Discovery (Removed)](#11-tflite-discovery-removed)

---

## 1. Overview

The `gen_config` package discovers which hardware executor (NSS/CSS/Host) each
layer of a model should run on. It produces two JSON files:

- **Report JSON** — complete discovery data for humans and tools
- **Compiler JSON** — minimal assignments consumed by the C++ `ExecutorAssignmentPass`

The system has a single entry path — the standalone CLI:

```
CLI (cli.py) ───(in-process)───► _runner.py ───► torq.lab (compile/run/compare)
                                      ├── _options.py   (DiscoveryConfig)
                                      ├── _cache.py     (versioned artifact cache)
                                      ├── _state.py
                                      ├── _report.py
                                      ├── ONNX helpers in torq.lab
                                      └── core.py
```

`discover` and `run` call `_runner.run_discovery()` /
`_runner.run_full_model()` in-process, so the CLI runs standalone from an
installed wheel with no source checkout. The standalone engine is the only
path, and discovery is ONNX-only.

---

## 2. Module Map

```
python/torq/gen_config/
├── __init__.py          Package init; exposes the CLI `main` lazily (PEP 562
│                        __getattr__) so submodule imports stay cheap.
├── __main__.py          Entry for `python -m torq.gen_config`
│
├── cli.py               CLI entry point (argparse; discover/run call
│                        _runner.py in-process)
├── view.py              Human-readable report viewer (standalone script)
│
├── core.py              Shared utilities: JSON I/O, recommendation, tolerance,
│                        timing, report computation.
├── _utils.py            MLIR parsing, diff metrics, table formatting.
├── _utils_mac.py        Per-layer MAC count computation (ONNX), plus a TFLite
│                        helper kept for the torq.testing shim.
│
├── _state.py            ExecutorDiscoveryState class + global singleton.
│                        Accumulates results during a discovery run.
├── _report.py           Report generation: sections, final report, JSON persistence.
│                        Format-agnostic — works with any ExecutorDiscoveryState.
│
├── _options.py          DiscoveryConfig dataclass: every discover/run option
│                        as a plain typed field.
│                        from_argparse() maps the CLI namespace onto it.
├── _cache.py            Content-versioned artifact cache. Root is always
│                        absolute (tools run with a different CWD).
└── _runner.py           Standalone engine: case generation, skip/dedup,
                         compile+run+compare loop, incremental JSON, reports.
                         Driven by cli.py.

    (ONNX layer/subgraph extraction, ONNX dtype conversions, ONNX
     quantization, the TFLite FlatBuffer layer extractor, the Case
     container and the verbosity flag live in torq.lab: the Case container at
     the package root, model_tools/extraction/onnx/,
     model_tools/dtype_conversion/onnx.py, quantization/onnx/,
     model_tools/extraction/tflite/, logging.py —
     re-exported by torq.testing for the in-tree test framework.)
```

### Responsibility split

| Module | Concern | Format-specific? |
|--------|---------|:---:|
| `core.py` | JSON I/O, recommendation, tolerance | No |
| `_utils.py` | MLIR line numbers, diff parsing, table formatting | No |
| `_utils_mac.py` | MAC counts per layer | **Yes** (ONNX) |
| `_state.py` | Accumulate discovery results in memory | No |
| `_report.py` | Generate human-readable reports from state | No |
| `_options.py` | `DiscoveryConfig` option container | No |
| `_cache.py` | Versioned artifact cache | No |
| `_runner.py` | Standalone discovery/run engine | **Yes** (ONNX) |
| `torq.lab` (`model_tools.extraction.onnx`, `model_tools.dtype_conversion.onnx`, `quantization.onnx`, `model_tools.extraction.tflite`, `logging`, package-root `Case`) | ONNX extraction/conversion/quantization, TFLite layer extraction, `Case`, verbosity | **Yes** (ONNX) |
| `cli.py` | argparse, user-facing commands (in-process) | No |
| `view.py` | Pretty-print report / compiler JSONs | No |

### Key helpers in `core.py`

| Function | Purpose | Used by |
|----------|---------|---------|
| `_build_report_from_ops(ops)` | Shared report computation: given an `ops` dict, returns `(summary, critical_failures, rows)`. Used by both live discovery (`_report.py`) and JSON editing (`cli.py`, via `generate_final_report_text()`) without circular imports. | `_report.py`, `cli.py` |

---

## 3. Dependency Graph

```
                        ┌──────────┐
                        │ _utils   │
                        └────┬─────┘
                             │
                   ┌─────────┴──────────┐
                   │                    │
              ┌────▼───┐           ┌───▼────┐
              │  core  │           │  view  │
              └───┬────┘           └────────┘
                  │
      ┌───────────┼───────────┐
      │           │           │
  ┌───▼───┐  ┌────▼───┐  ┌────▼─────────┐     ┌─────────┐ ┌────────┐
  │ _state│  │_report │  │    _runner   │◄────│ _options │ │ _cache │
  └───┬───┘  └───┬────┘  └───────┬──────┘     └─────────┘ └────────┘
      │          │               │  (also: _utils_mac, torq.lab)
      │          │       ┌────────┴───────┐
      │          │       │                │
      │          │  ┌────▼─────┐     ┌────▼─────┐
      │          │  │lab.quant │     │ lab.onnx │
      │          │  │ize_onnx  │     └──────────┘
      │          │  └──────────┘
      └────┬─────┴───┴────┬────┘
           │              │
      ┌────▼──────────────▼──────┐
      │          cli.py          │ ◄── torq-gen-config /
      └──────────────────────────┘     python -m torq.gen_config
                                        (imports _runner lazily, in-process)
```

**Rules:**

1. `core.py` depends only on `_utils.py` and stdlib — it has no
   model-format dependencies. One import from `_utils` is deferred (inside a
   function) to avoid a circular dependency.
2. `_runner.py` holds the discovery/run engine; the ONNX-specific extraction
   and conversion live in `torq.lab.model_tools.extraction.onnx` / `torq.lab.model_tools.dtype_conversion.onnx`,
   quantization in `torq.lab.quantization.onnx.static`.
3. No module in the package may import the test framework (`torq.testing`)
   — this is enforced by `scripts/verify_torq_wheel.sh`, which checks the
   packaged modules for test-framework imports and asserts that importing
   the CLI does not pull them into `sys.modules`.
4. `torq.lab.quantization.onnx.static` is the quantization helper used by `_runner.py`
   and `cli.py`.
5. `view.py` is an internal utility that imports from `_utils`; it is not
   part of the public API (the CLI is the public surface).
6. `torq.testing` keeps thin compatibility shims (`quantize_onnx.py`,
   `tflite_layer_extractor.py`, `cases.py`) that
   re-export from `torq.lab`; nothing imports the other way.
7. The TFLite FlatBuffer layer extractor lives in `torq.lab.model_tools.extraction.tflite` as a
   utility (used by the `_utils_mac.py` TFLite MAC helper and the
   `torq.testing` shim); it plays no role in discovery.

---

## 4. Two-JSON Design

| File | Format | Audience | Contains |
|------|--------|----------|----------|
| `torq_gen_config_<model>.json` | Report | Humans, tools, `edit` command | `ops`, `discovery_report`, statuses, tolerances, timing, `final_report_text` |
| `torq_gen_config_<model>_compiler.json` | Compiler | C++ `ExecutorAssignmentPass` | `op_assignments`, `model_name` |

### Report JSON (`ops` format)

```json
{
  "version": "1.1",
  "model_name": "encoder",
  "default_tolerance": {"fp_avg_tol": 0.01, "fp_max_tol": 0.01},
  "ops": {
    "Conv_conv_out": {
      "executors": {
        "nss": {"status": "error", "failure_report": {...}},
        "css": {"status": "success", "timing": {"runtime_ms": 12.3}},
        "host": {"status": "success", "timing": {"runtime_ms": 45.6}}
      },
      "recommended_executor": "css",
      "mlir_location": "10:10",
      "_node_index": 0
    }
  },
  "discovery_report": {
    "summary": {...},
    "critical_failures": [...]
  },
  "final_report_text": "FINAL EXECUTOR DISCOVERY REPORT\n..."
}
```

### Compiler JSON (`op_assignments` format)

```json
{
  "op_assignments": {
    "10:10": {"executor": "css"},
    "11:10": {"executor": "host"}
  },
  "model_name": "encoder"
}
```

### Generation rules

- `discover` writes both JSONs at the end of each layer test
- `edit` reads the report JSON, updates it, then regenerates the compiler JSON
- `run` reads either JSON: if report exists, regenerates compiler from it;
  if only compiler exists, uses it directly
- The C++ pass accepts both formats: `op_assignments` with `line:col` keys, or `ops`
  with `mlir_location` + `recommended_executor`

---

## 5. Quantization Support

Quantization is implemented in `torq.lab.quantization.onnx.static`. It takes an FP32 ONNX model and produces either QDQ (Quantize-Dequantize) or QOperator (native int8 ops such as `QLinearConv`) format.

### Discovery flow with quantization

```
_user passes --quantize --full-integer --quant-format=qdq_
_runner.py:_maybe_apply_quantization()
    ├── quantize the full model/subgraph once
    ├── import the quantized ONNX into MLIR
    ├── build mapping: original layer op-type → quantized compute-op line
    │   (strips Q/DQ wrappers, aligns via LCS)
    └── quantize each layer individually for per-layer tests
```

### Mapping quantized MLIR to original layers

The C++ `ExecutorAssignmentPass` must receive line numbers from the **final** quantized MLIR, not the original FP32 MLIR. The mapping therefore:

1. Imports the quantized ONNX model into MLIR.
2. Discards `QuantizeLinear`/`DequantizeLinear` wrapper ops.
3. Aligns the remaining compute ops with the original layer order using **Longest Common Subsequence (LCS)**.
   - Fused activations (e.g. `Conv`+`Relu`) disappear in the quantized MLIR and are skipped by LCS.
   - Each surviving compute op keeps its correct quantized MLIR line number.
4. Stores `node_index` as an index into the compute-only op list.

`_runner.py:_update_discovery_json_line_numbers()` uses the same compute-only list when resolving `_node_index`, and falls back to op-type matching (or clears stale locations) for fused/skipped layers.

### Quantization options

| Option | Effect |
|--------|--------|
| `--quantize` | Run ONNX Runtime static quantization on each layer and on the full model |
| `--per-channel` | Use per-channel weight quantization |
| `--full-integer` | Convert input Q/output DQ to int8 I/O (remove graph input/output Q/DQ) |
| `--quant-format=qdq` | Insert `QuantizeLinear`/`DequantizeLinear` nodes around ops (default) |
| `--quant-format=qoperator` | Use native int8 ops such as `QLinearConv` |

**Note:** `discover` and `run` must use the same quantization flags so the full model compiler JSON matches the final quantized MLIR.

See [`onnx_quantization.md`](onnx_quantization.md) for the
full quantization design, mapping algorithm, and CLI examples.

## 6. CLI Subcommand Flow

### `discover`

```
User: torq-gen-config discover --model model.onnx ...

cli.py:cmd_discover()
    ├── _validate_model_and_flags() (model exists, ONNX, flag exclusivity)
    ├── DiscoveryConfig.from_argparse(args)
    └── _runner.run_discovery(cfg)                     (in-process)
        ├── _generate_test_cases() → layer × executor matrix
        │     (BF16/INT32/quantize conversion, subgraph extraction, dedup)
        ├── for each chip resolved from --torq-hw, for each case:
        │     skip/dedup checks → compile (torq.lab.ModelPipeline)
        │     → run → compare vs reference → record into _state
        │     → _save_discovery_results() after every case (both JSONs)
        └── _finalize_report() → save + print final report
```

### `run`

```
User: torq-gen-config run --model model.onnx ...

cli.py:cmd_run()
    ├── Verify report or compiler JSON exists
    ├── DiscoveryConfig.from_argparse(args)
    └── _runner.run_full_model(cfg)                    (in-process)
        ├── _resolve_executor_assignments():
        │     ├── _find_discovery_json() or _find_compiler_json()
        │     ├── _update_discovery_json_line_numbers() (report only)
        │     ├── generate_compiler_config()
        │     └── versioned copy → passed to C++ via --torq-executor-map
        └── full-model compile → run → capture-stdout compare
            → parse metrics → report
```

### `view`

```
User: torq-gen-config view torq_gen_config_model.json [layer_id]

cli.py:cmd_view()
    ├── Load JSON
    ├── If layer_id: print_layer_details()
    └── Else: print_summary()
```

### `edit`

```
User: torq-gen-config edit --model model.onnx --layer Conv_conv_out --executor nss

cli.py:cmd_edit()
    ├── _resolve_edit_path() → from --model or positional config path
    ├── _detect_compiler_json() → guard against accidental compiler JSON
    ├── Match layers via --layer (exact, substring, fnmatch `*`|`?`, ALL)
    ├── Update recommended_executor and/or tolerance
    ├── Regenerate final_report_text via generate_final_report_text()
    ├── save_config(report_path)
    └── generate_compiler_config() → save_config(compiler_path)

**Layer matching order:**
1. `ALL` → every layer
2. Contains `*` or `?` → fnmatch pattern
3. Exact case-insensitive match → that layer only
4. Fallback → substring match (all layers containing the query)
```

---

## 7. How to Add a New Model Format

> **Note:** This branch focuses on ONNX models. The pattern below shows how a
> new format could be added by creating format-specific modules that reuse the
> same discovery algorithm. New formats should put the cores in a
> `_runner_<format>.py` (or format-specific functions inside `_runner.py`),
> mirroring how the ONNX flow is split between `_runner.py` and `torq.lab.model_tools.extraction.onnx`.

### Step 1: Create `_<format>.py` extraction/conversion helpers

Mirror `torq.lab.model_tools.extraction.onnx` / `torq.lab.model_tools.dtype_conversion.onnx`: model loading,
layer/subgraph extraction, and conversion to MLIR:

```python
# _<format>.py
"""<Format> layer/subgraph extraction and model conversion."""

from torq.lab import Case


def generate_<format>_layers_from_file(...) -> List[Case]:
    """Extract one Case per layer (layer model + node index)."""
    ...


def convert_<format>_to_mlir(model_path, out_dir) -> Path:
    """Convert a (layer) model to MLIR via the toolchain."""
    ...


def build_<format>_to_mlir_mapping(model_path, mlir_file):
    """Map <format> ops to full-model MLIR line:column locations."""
    ...
```

### Step 2: Add the engine hooks in `_runner.py` (or `_runner_<format>.py`)

```python
def _generate_test_cases_<format>(cfg, cache):
    """Generate (case, layer_id, executor, ...) tuples — same matrix as ONNX."""
    ...
```

Reuse the shared machinery as-is: skip/dedup, the versioned cache drivers,
`_compile_model()` / `_make_compare_fn()`, result recording into `_state.py`,
and the `_report.py` final report — none of it is format-specific beyond the
case tuple contents.

### Step 3: Wire the format into `cli.py`

Extend `_validate_model_and_flags()` to accept the new extension and dispatch
to the format's `run_discovery()` / `run_full_model()` cores.

### Key principle

The discovery algorithm is format-agnostic in its structure — it compiles a
layer, runs it, compares results, records status. Only the model loading,
MLIR generation, and case parametrization are format-specific.

---

## 8. How to Add a New CLI Subcommand

### Step 1: Add the handler in `cli.py`

```python
def cmd_compare(args: argparse.Namespace) -> int:
    """Compare two discovery JSONs and show differences."""
    data_a = load_config(Path(args.config_a))
    data_b = load_config(Path(args.config_b))
    # ... comparison logic ...
    return 0
```

### Step 2: Register in `main()`

```python
compare_parser = subparsers.add_parser("compare", help="Compare two configs")
compare_parser.add_argument("config_a", help="First config JSON")
compare_parser.add_argument("config_b", help="Second config JSON")
compare_parser.set_defaults(func=cmd_compare)
```

### Step 3: Add tests

```python
# tests/test_gen_config_cli.py
def test_compare(self):
    ...
```

### Convention

- Handler functions are named `cmd_<subcommand>`
- They return `int` (0 = success, 1 = error)
- They print errors to `sys.stderr`
- Validation happens early (file exists, JSON is valid) before logic

---

## 9. Testing Architecture

### Test files

| File | What it tests | Runs as |
|------|--------------|---------|
| `tests/test_gen_config_cli.py` | Quantize helpers (module-level unit tests) + standalone CLI workflows (`TestStandaloneCliWorkflows`): one shared host-only discovery through the standalone engine, then `edit`/`view`/`run` against its JSON output | Unit tests run by default; workflows are opt-in via `-m gen_config_standalone` |

### CLI test pattern

Each CLI workflow test:
1. Creates a temp directory
2. Invokes the CLI exactly as users would (`python -m torq.gen_config ...`
   in a subprocess)
3. Validates the generated JSON (structure, fields, content)
4. Cleans up temp directory + any generated JSONs

`scripts/verify_torq_wheel.sh` complements these as the packaging guard: it
installs the compiler wheel into a clean venv, exercises the
`torq-gen-config` console script, and asserts that no packaged
`torq/gen_config` module imports the test framework (and that importing the
CLI does not pull it into `sys.modules`).

### Running tests

```bash
# Fast unit tests (default)
pytest tests/test_gen_config_cli.py -v

# Standalone end-to-end CLI workflows (opt-in; runs a real discovery through
# the standalone engine)
pytest tests/test_gen_config_cli.py -m gen_config_standalone -v
```

---

## 10. Standalone CLI Options

Every discover/run option is a native `torq-gen-config` flag
(`python3 -m torq.gen_config` is equivalent).

| Option | Description |
|--------|-------------|
| `--model` | Path to ONNX model |
| `--output-dir` | Directory for executor config JSON (default: current directory) |
| `--skip-mode` | Stop after first success per layer |
| `--recompute-cache` | Force recompute (ignore cache) |
| `--debug-ir=DIR` | Dump IR for debugging |
| `--skip-executors=nss,css` | Skip specific executors |
| `--skip-ops=MaxPool,Add` | Skip specific ONNX op types |
| `--auto-convert-bf16` | Convert FP32 to BF16 |
| `--auto-convert-int32` | Convert INT64 tensors to INT32 |
| `--save-bf16-model=PATH` | Save converted BF16 model |
| `--subgraph-from=OP` | Subgraph start |
| `--subgraph-to=OP` | Subgraph end |
| `--collect-timing` | Collect compile and runtime timing data |
| `--timing-runs=N` | Number of runtime runs for timing average (default: 1) |
| `--recommend-by-timing` | Recommend fastest executor based on timing data |
| `--log-file=PATH` | Redirect all output to log file |
| `-v`, `--verbose` | Show detailed logs |
| `--dedup-layers` | Detect duplicate layers and copy results |
| `--quantize` | Quantize layers/full model to int8 |
| `--per-channel` | Per-channel weight quantization |
| `--full-integer` | Rewrite quantized I/O to int8 |
| `--quant-format` | ONNX quantization format: `qdq` (default), `qoperator`, or `hybrid` |
| `--torq-hw TARGET` | Target hardware for `torq-compile --torq-hw`: a target such as SL2610, or a chip name/group from `tests/testdata/chips` (default: `default` = SL2610) |
| `--torq-hw-type TYPE` | Runtime hardware type passed to `torq-run-module --torq_hw_type` (default: `sim`) |
| `--compiler-option OPT` | Extra option passed through to `torq-compile` (repeatable) |
| `--runtime-option OPT` | Extra option passed through to `torq-run-module` (repeatable) |

```bash
# Layer discovery with skip mode
torq-gen-config discover --model model.onnx --skip-mode

# Full model with debug output
torq-gen-config run --model model.onnx --debug-ir=tmp

# Subgraph debugging
torq-gen-config discover --model model.onnx --subgraph-from=StartOp --subgraph-to=EndOp

# Skip crashing executors
torq-gen-config discover --model model.onnx --skip-executors=nss

# Timing-based executor recommendation
torq-gen-config discover --model model.onnx --collect-timing --timing-runs=5 --recommend-by-timing

# Redirect output to log file
torq-gen-config discover --model model.onnx --log-file=discovery.log

# Skip duplicate layers
torq-gen-config discover --model model.onnx --dedup-layers --skip-mode

# Re-test a single layer (scope a subgraph to that one op)
torq-gen-config discover --model model.onnx \
    --subgraph-from=Conv_0 --subgraph-to=Conv_0 \
    --recompute-cache
```

---

## 11. TFLite Discovery (Removed)

TFLite discovery has been removed; `torq-gen-config` is ONNX-only. Passing a
`.tflite` model to `discover` / `run` is rejected with an error.

| Model extension | Entry point |
|-----------------|------------------|
| `.onnx` | `torq-gen-config discover` / `run` |
| `.tflite` | not supported |

`view` and `edit` still work on previously generated TFLite report JSONs —
they operate on the generic `ops` format, which is format-agnostic. TFLite
layers are keyed by operator name and index: `{OP_NAME}_{op_index}` (e.g.
`CONV_2D_0`, `DEQUANTIZE_1`), and those IDs are used everywhere a layer ID is
expected (`view`, `edit`).

The FlatBuffer layer extractor (now `torq.lab.model_tools.extraction.tflite`) remains available as a
utility (TFLite MAC counting, `torq.testing` compatibility shims) but plays
no role in discovery.
