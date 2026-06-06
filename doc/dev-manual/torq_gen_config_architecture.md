# Gen-Config Architecture

---

## Table of Contents

1. [Overview](#1-overview)
2. [Module Map](#2-module-map)
3. [Dependency Graph](#3-dependency-graph)
4. [Two-JSON Design](#4-two-json-design)
5. [CLI Subcommand Flow](#5-cli-subcommand-flow)
6. [How to Add a New Model Format](#6-how-to-add-a-new-model-format)
7. [How to Add a New CLI Subcommand](#7-how-to-add-a-new-cli-subcommand)
8. [Testing Architecture](#8-testing-architecture)

---

## 1. Overview

The `gen_config` package discovers which hardware executor (NSS/CSS/Host) each
layer of a model should run on. It produces two JSON files:

- **Report JSON** — complete discovery data for humans and tools
- **Compiler JSON** — minimal assignments consumed by the C++ `ExecutorAssignmentPass`

The system is split into three layers:

```
CLI (cli.py) ───(subprocess)───► pytest ───► discovery.py
                                                  ├── _state.py
                                                  ├── _report.py
                                                  ├── _cases.py
                                                  ├── _utils.py
                                                  └── core.py
```

---

## 2. Module Map

```
python/torq/gen_config/
├── __init__.py          Empty (all imports go through discovery.py)
├── __main__.py          Entry for `python -m torq.gen_config`
│
├── cli.py               CLI entry point (argparse + subprocess orchestration)
├── view.py              Human-readable report viewer (standalone script)
│
├── core.py              Shared utilities: JSON I/O, recommendation, tolerance,
│                        timing, report computation. No pytest dependency.
├── _utils.py            MLIR parsing, diff metrics, table formatting.
│
├── _state.py            ExecutorDiscoveryState class + global singleton.
│                        Accumulates results during a pytest session.
├── _report.py           Report generation: sections, final report, JSON persistence.
│                        Format-agnostic — works with any ExecutorDiscoveryState.
├── _cases.py            ONNX-specific: pytest fixtures, test parametrization,
│                        executor_discovery(), BF16 conversion, subgraph extraction.
│
├── discovery.py         PUBLIC ENTRY POINT. Re-exports from _state, _report,
│                        and _cases. Everything outside gen_config/ imports
│                        from here.
│
└── pytest_plugin.py     Pytest hooks: option registration, log redirection,
                         terminal progress, error recording.
```

### Responsibility split

| Module | Concern | Format-specific? |
|--------|---------|:---:|
| `core.py` | JSON I/O, recommendation, tolerance | No |
| `_utils.py` | MLIR line numbers, diff parsing, table formatting | No |
| `_state.py` | Accumulate discovery results in memory | No |
| `_report.py` | Generate human-readable reports from state | No |
| `_cases.py` | ONNX fixtures, hooks, test generation | **Yes** |
| `cli.py` | argparse, subprocess, user-facing commands | No |
| `view.py` | Pretty-print report / compiler JSONs | No |
| `discovery.py` | Public re-export facade | No |
| `pytest_plugin.py` | Pytest option registration, log redirection | No |

### Key helpers in `core.py`

| Function | Purpose | Used by |
|----------|---------|---------|
| `_opt(config, short, legacy, default)` | Resolve pytest option aliases (canonical name first, then legacy fallback). Enables smooth option renaming without breaking existing CLI invocations. | `pytest_plugin.py`, `_cases.py` |
| `_build_report_from_ops(ops)` | Shared report computation: given an `ops` dict, returns `(summary, critical_failures, rows)`. Used by both live discovery (`_report.py`) and JSON editing (`cli.py`) without circular imports. | `_report.py`, `cli.py` |

---

## 3. Dependency Graph

```
                        ┌──────────┐
                        │ _utils   │
                        └────┬─────┘
                             │
                   ┌─────────┼──────────┐
                   │         │          │
              ┌────▼───┐  ┌─▼────┐  ┌──▼────┐
              │  core  │  │ view │  │pytest_│
              └───┬────┘  └──────┘  │plugin │
                  │                 └───┬───┘
      ┌───────────┼───────────┐       │
      │           │           │       │
  ┌───▼───┐  ┌────▼───┐  ┌────▼──────▼───┐
  │ _state│  │_report │  │     _cases     │ ← ONNX-specific
  └───┬───┘  └───┬────┘  └───────┬────────┘
      │          │               │
      └────┬─────┴───────────────┘
           │
      ┌────▼────────────┐
      │   discovery.py  │ ← public facade
      └────┬────────────┘
           │
      ┌────▼────────────┐
      │ external callers│
      │ test_onnx_gen_  │
      │ config.py       │
      └─────────────────┘
```

**Rules:**

1. `core.py` depends only on `_utils.py` and stdlib — it has no pytest or
   model-format dependencies. One import from `_utils` is deferred (inside a
   function) to avoid a circular dependency.
2. `_report.py` does **not** import from `_cases.py` — no circular dependency.
3. `_cases.py` is the only format-specific module. It imports from
   `_state`, `_report`, `core`, and `_utils`.
4. `discovery.py` is a pure re-export facade. It imports from `_state`,
   `_report`, and `_cases`. No logic lives here.
5. Everything outside `gen_config/` imports from `discovery.py` only.
   No external code imports `_state`, `_report`, or `_cases` directly.
6. `view.py` and `pytest_plugin.py` are internal utilities that import from
   `_utils` (and `core` for the plugin). They are not part of the public API.

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

## 5. CLI Subcommand Flow

### `discover`

```
User: torq-gen-config discover --model model.onnx ...

cli.py:cmd_discover()
    ├── Map CLI flags → pytest extra_args
    └── subprocess: pytest tests/test_onnx_gen_config.py -k "_layer_"

        pytest loads pytest_plugin.py → registers options, sets up logging
        discovery.py:pytest_generate_tests() → generates layer × executor matrix
        For each test:
            executor_discovery() → compile → run → compare → record_result()
        save_progress fixture → _save_discovery_results() → writes both JSONs
        pytest_sessionfinish → _print_final_report() → terminal + log file
```

### `run`

```
User: torq-gen-config run --model model.onnx ...

cli.py:cmd_run()
    ├── Verify report or compiler JSON exists
    └── subprocess: pytest tests/test_onnx_gen_config.py -k "_full_model"

        torq_torq_gen_config_json fixture:
            ├── _find_discovery_json() or _find_compiler_json()
            ├── _update_discovery_json_line_numbers() (report only)
            ├── generate_compiler_config()
            └── Write to versioned_file → passed to C++ via --torq-executor-map

        Full model compiles with executor assignments, runs, compares
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

## 6. How to Add a New Model Format

Example: adding Torch model support.

### Step 1: Create `_cases_torch.py`

Copy the structure of `_cases.py` but replace ONNX-specific logic:

```python
# _cases_torch.py
"""Torch executor discovery test cases and fixtures."""

import torch
import pytest

from torq.gen_config._state import ExecutorDiscoveryState, _discovery_state
from torq.gen_config.core import (
    DEFAULT_TOLERANCE, EXECUTOR_ORDER, _discovery_log,
    _get_json_path, _load_json, _opt,
    build_timing_data, get_tolerance, ...
)
from torq.gen_config._report import _get_all_critical_failures, _save_detailed_report

# Torch-specific model loading
from torq.testing.torch import load_torch_model, generate_torch_layers


def _discover_model_files(config):
    """Discover Torch model files from --model-path."""
    # Return .pt or .pth files instead of .onnx
    ...


def _build_torch_to_mlir_mapping(model_path, mlir_file):
    """Build mapping from Torch ops to MLIR line numbers."""
    # Torch-specific MLIR import
    ...


def _generate_layer_cases(f, config):
    """Generate test cases in Torch layer-extraction mode."""
    # Use generate_torch_layers instead of generate_onnx_layers_from_file
    ...


def pytest_generate_tests(metafunc):
    """Generate test cases for each layer."""
    # Same pattern as _cases.py but with Torch-specific functions
    ...


@pytest.fixture
def torch_layer_model(request, layer_executor_case):
    """Provide the Torch layer model."""
    # Torch-specific fixture
    ...


def executor_discovery(request, torq_results, reference_results,
                       case_config, layer_executor_case, torch_mlir_model_file,
                       discovery_state=_discovery_state):
    """Core executor discovery — same algorithm, different model format."""
    # Same algorithm as _cases.py:executor_discovery()
    ...
```

### Step 2: Add re-exports to `discovery.py`

```python
# discovery.py (add these lines)

# Re-export Torch case generation and fixtures
from torq.gen_config._cases_torch import (
    executor_discovery as executor_discovery_torch,
    pytest_generate_tests as pytest_generate_tests_torch,
    torch_layer_model,
    ...
)
```

### Step 3: Create the test entry point

```python
# tests/test_torch_gen_config.py
"""Thin orchestration — same pattern as test_onnx_gen_config.py."""

from torq.gen_config.discovery import pytest_generate_tests_torch as pytest_generate_tests
from torq.gen_config.discovery import (
    executor_discovery_torch as executor_discovery,
    torch_layer_model as onnx_layer_model,
    reference_results,
    ...
)
```

### Key principle

The discovery algorithm (`executor_discovery()`) is format-agnostic in its
structure — it compiles a layer, runs it, compares results, records status.
Only the model loading, MLIR generation, and case parametrization are
format-specific. These live in `_cases_<format>.py`.

---

## 7. How to Add a New CLI Subcommand

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

## 8. Testing Architecture

### Test files

| File | What it tests | Runs as |
|------|--------------|---------|
| `tests/test_gen_config_cli.py` | All CLI options end-to-end | Subprocess (spawns pytest) |
| `tests/test_onnx_gen_config.py` | Discovery orchestration | Direct (imports from `discovery.py`) |

### CLI test pattern

Each CLI test:
1. Creates a temp directory
2. Calls `_run_pytest_and_validate()` which spawns `pytest test_onnx_gen_config.py`
3. Validates the generated JSON (structure, fields, content)
4. Cleans up temp directory + any generated JSONs

### Adding tests for a new format

Follow the same pattern as `test_gen_config_cli.py`:

```python
TEST_MODEL = PROJECT_ROOT / "tests/testdata/torch_models/example_gen_config.pt"

class TestTorchGenConfigIntegration:
    def test_basic_discovery(self):
        ...
```

### Running tests

```bash
# All CLI integration tests
pytest tests/test_gen_config_cli.py -v

# Specific test
pytest tests/test_gen_config_cli.py -k "test_compiler_json" -v
```
