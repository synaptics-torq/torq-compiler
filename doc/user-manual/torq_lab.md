# torq.lab

`torq.lab` is a generic compile / run / profile orchestration layer around `torq-compile` and `torq-run-module`. It wraps the artifact layout, input/output handling, local **and** remote (SSH/ADB) execution, output comparison, and profiling into a single reusable pipeline.

It is available via two interfaces:

- the **`torq-lab` command line** — a thin wrapper for compiling, running, and comparing a model from the shell; and
- the **`torq.lab` Python package** — a pytest-independent library you can import into your own scripts and test harnesses.

```{note}
`torq.lab` is bundled in the `torq-compiler` wheel (release 2.0.0 and above), so it is available wherever `torq-compile` / `torq-run-module` are.
```

---

## Installation

`torq.lab` is installed as part of the `torq-compiler` wheel — see [Quickstart → Python Wheel](getting_started.md). The `torq-lab` console script and `python -m torq.lab` entry point are available immediately after install.

Some optional features pull in extra dependencies:

```bash
# Profiling annotation + Perfetto trace rendering
# Adds `pandas`, `XlsxWriter`, and `protobuf` dependencies
$ pip install "torq_compiler-<version>-<platform>.whl[profile]"

# ONNX reference implementations / decoder extraction helpers
$ pip install "torq_compiler-<version>-<platform>.whl[onnx]"
```

Without these dependencies, relevant helpers such as in `torq.lab.profiling` raise a clear `LabError` when the extra is missing.

---

## Command line

```bash
$ torq-lab <command> <model> [options]
# equivalently:
$ python -m torq.lab <command> <model> [options]
```

`<model>` is a `.mlir` source for `compile` / `compile-run` / `compare`, and a `.vmfb` module for `run`. It is optional when `--config` supplies it.

### Subcommands

| Command       | Purpose                                                        |
|---------------|----------------------------------------------------------------|
| `compile`     | Compile an MLIR model to a VMFB.                               |
| `run`         | Run a compiled VMFB.                                           |
| `compile-run` | Compile an MLIR model and run it in one step.                 |
| `compare`     | Compile, run, and check outputs against `--expected-output-npy`. |

### Examples

Compile an MLIR file to a VMFB:

```bash
$ torq-lab compile tests/testdata/tosa_ops/add.mlir -o model.vmfb
```

Compile and run against the simulator with random inputs:

```bash
$ torq-lab compile-run tests/testdata/tosa_ops/add.mlir --random-inputs
```

Run a pre-compiled module with explicit inputs:

```bash
$ torq-lab run model.vmfb --input-npy in0.npy --input-npy in1.npy
```

Compile, run, and verify outputs against a golden reference:

```bash
$ torq-lab compare model.mlir \
    --random-inputs \
    --expected-output-npy golden_0.npy
```

### Choosing where the model runs

`--runtime-hw-type` selects the execution target. It also steers the compile: the pipeline adds the matching `torq-compile` host flags for you, so the same command line works for the simulator and for a board.

| `--runtime-hw-type` | Runs on                                   | Compile flags added                                       |
|---------------------|-------------------------------------------|-----------------------------------------------------------|
| `sim` (default)     | the cmodel and mpact simulation           | `--torq-target-host-triple=native`                        |
| `aws_fpga`          | an AWS FPGA instance                      | `--torq-target-host-triple=native`                        |
| `astra_machina`     | an Astra Machina board                    | none — the default (board) target triple is used          |

A cmodel run is therefore the zero-configuration default:

```bash
$ torq-lab compile-run model.mlir --random-inputs
```

Running on a board additionally needs an address, so the pipeline can stage the artifacts and execute there:

```bash
$ torq-lab compile-run model.mlir --random-inputs \
    --runtime-hw-type astra_machina \
    --remote <board IP/ADB address>
```

See [Remote execution](#remote-execution) for the SSH and ADB address forms.

### Common options

| Option                     | Description                                                        |
|----------------------------|--------------------------------------------------------------------|
| `--config PATH`            | Config JSON file(s) to layer (repeatable, later wins). Alternative to the flags below. |
| `--work-dir DIR`           | Artifact directory (default: `./<model-stem>-run`).               |
| `--chip CHIP`              | Chip target for `--torq-hw` (default: `SL2610`).                  |
| `--runtime-hw-type TYPE`   | Where to run: `sim`, `aws_fpga`, `astra_machina`, … (see [above](#choosing-where-the-model-runs)). |
| `--function NAME`          | Entry function name. Rarely needed as it is parsed from the MLIR automatically (falling back to `main`), so set it only to pick a specific function out of a multi-function module. |
| `--compiler-option OPT`    | Extra `torq-compile` flag (repeatable).                           |
| `--runtime-option OPT`     | Extra `torq-run-module` flag (repeatable).                        |
| `--input-npy PATH`         | Input `.npy` file (repeatable).                                   |
| `--random-inputs`          | Generate random inputs from the MLIR IO spec.                     |
| `--expected-output-npy P`  | Expected output `.npy` for comparison (repeatable).               |
| `--dump-ir`                | Dump IR after each pass into `<work-dir>/debug/ir`.               |
| `--dump-phases`            | Dump compilation phases into `<work-dir>/phases`.                 |
| `--profile-compile`        | Enable compile-time profiling.                                    |
| `--profile-runtime`        | Enable runtime host profiling.                                    |
| `-o, --output PATH`        | Output VMFB path (`compile` only; default `<work-dir>/model.vmfb`). |
| `--timeout SECONDS`        | Per-tool timeout in seconds — see the note below.                 |

Remote options are described under [Remote execution](#remote-execution).

`--timeout` is a single value applied *independently* to each tool invocation, not a budget for the whole pipeline: `compile-run` gives the compile its own `SECONDS` and the run another `SECONDS`. There is currently no way to set a different limit for the compiler and the runtime. On a remote run the same value also bounds each individual command issued over the SSH/ADB transport (default `15` when `--timeout` is unset).

### Exit codes

| Code | Meaning                                                             |
|------|---------------------------------------------------------------------|
| `0`  | Success (and, for `compare`, outputs matched).                      |
| `1`  | Error — a tool failed, or `compare` had no expected outputs (from neither `--expected-output-npy` nor a config). |
| `2`  | Comparison failed — the run's outputs did not match the expected ones. |

A run that produced no outputs (or a mismatched output count) while expected outputs were given is reported as a mismatch (exit `2`), not a pass.

Every invocation writes a `manifest.json` into the work dir (see [The work directory and manifest](#the-work-directory-and-manifest)).

---

## Config files and layering

Instead of a long flag list you can supply one or more JSON config files with `--config`. Files are deep-merged in order (later wins), docker-compose style: keep a shared base and overlay per-environment differences.

```json
// base.json
{
  "model_path": "model.mlir",
  "chip": "SL2610",
  "compiler_options": ["--some-flag"],
  "random_inputs": true
}
```

```json
// board.json — overlay adding a remote target
{
  "runtime_hw_type": "astra_machina",
  "remote": { "address": "root@10.46.130.17" }
}
```

`remote.address` takes the same forms as `--remote`, so an ADB-attached board is configured the same way — `"adb"` for the first attached device, or an explicit serial:

```json
// board-adb.json — the same board over ADB instead of SSH
{
  "runtime_hw_type": "astra_machina",
  "remote": { "address": "adb", "stage_runner": "./torq-run-module" }
}
```

```bash
$ torq-lab compile-run --config base.json --config board.json
```

Note that the config files you pass in and the `manifest.json` a run writes out are different directions of the same schema, not three competing formats: `base.json` and `board.json` are **inputs** you author, while `manifest.json` is an **output** the pipeline records (see [The work directory and manifest](#the-work-directory-and-manifest)). Because a manifest embeds the config it ran with, it can be fed straight back in as a layer — that is the only overlap.

Explicit positional/flag values still override the layered files (e.g. a positional `<model>` overrides `model_path`; `--remote` overrides the config's `remote.address`). Each source may be a bare config dict **or** a full `manifest.json` — its `config` section is used — so a saved manifest can seed a new run.

---

## Remote execution

`torq.lab` can stage the module and inputs onto a board, run `torq-run-module` there, and pull the outputs back. Enable it with `--remote` (or a `remote` section in a config file).

```bash
$ torq-lab compile-run model.mlir \
    --random-inputs \
    --runtime-hw-type astra_machina \
    --remote root@10.46.130.17 \
    --stage-runner ./torq-run-module
```

`--remote` accepts the same address forms as the transport factory:

- `adb` — the first attached ADB device;
- an ADB serial number;
- `user@host` or a bare hostname/IP — over SSH.

| Option                 | Description                                                  |
|------------------------|--------------------------------------------------------------|
| `--remote ADDR`        | Remote board address (see forms above).                     |
| `--remote-port PORT`   | SSH port (default: `22`).                                   |
| `--remote-key PATH`    | SSH private key path.                                       |
| `--remote-runner PATH` | Absolute path of `torq-run-module` already on the device.  |
| `--stage-runner PATH`  | Local `torq-run-module` to copy onto the device.           |

---

## Profiling

Passing `--profile-compile` and/or `--profile-runtime` turns on the profiling path. The compile always emits debug info, and (with the `[profile]` extra installed) the pipeline adds the relevant traces and renders a Perfetto viewer into `<work-dir>/profiles/`.

**Compile-time profiling** (`--profile-compile`) needs only a compile — no run. `torq-lab compile --profile-compile` writes `compile_profile.csv`, the compile Perfetto trace(s) (`*_compile.pb`), and `perfetto_viewer.html`:

```bash
$ torq-lab compile model.mlir --profile-compile
```

**Runtime profiling** (`--profile-runtime`) needs a run: the pipeline annotates the runtime host profile against the compile-time debug info and renders the viewer once the run completes.

```bash
$ torq-lab compile-run model.mlir --random-inputs --profile-runtime
```

The two are independent and can be combined (`compile-run --profile-compile --profile-runtime`), in which case the viewer merges the compile and runtime traces.

For the meaning of the trace columns and the annotated-profile format, see the [Performance Profiling Tool](profiling.md) chapter — `torq.lab` produces the same artifacts through the same underlying machinery.

---

## The work directory and manifest

Everything a run produces lands under `--work-dir` (default `./<model-stem>-run`):

```
<work-dir>/
├── model.vmfb          # compiled module
├── manifest.json       # config + results record (see below)
├── inputs/             # materialized .bin / .npy inputs
├── outputs/            # raw output_<i>.bin files
├── debug/              # --torq-debug-info (and ir/ under --dump-ir)
├── phases/             # --dump-phases
├── profiles/           # host_profile.csv, traces, perfetto_viewer.html
└── remote/             # scratch for staged remote runs
```

`manifest.json` (schema version 2) keeps two concerns separate:

- **`config`** — the pipeline inputs, round-trippable back into a `PipelineConfig` (and a `RemoteTarget`, when present). This lets a run be reconstructed from its manifest.
- **`results`** — what the run produced: the `compile` / `run` commands and timings, output paths, profiling artifacts, the `comparison` verdict, and any `diagnostics`.

It is written atomically (temp file + rename), so a concurrent reader always sees a complete file.

---

## Python library

The same pipeline is importable. The typical flow builds a `PipelineConfig`, drives a `ModelPipeline`, and inspects the structured results:

```python
from pathlib import Path

from torq.lab.pipeline import ModelPipeline
from torq.lab.types import PipelineConfig, RemoteTarget

config = PipelineConfig(
    model_path=Path("model.mlir"),
    work_dir=Path("model-run"),
    chip="SL2610",
    random_inputs=True,
    expected_output_npy=[Path("golden_0.npy")],
)

pipe = ModelPipeline(config)
compile_result = pipe.compile()          # -> CompileResult
run_result = pipe.run(compile_result.vmfb_path)  # -> RunResult
comparison = pipe.compare(run_result)    # -> ComparisonResult | None
pipe.write_manifest(
    compile_result=compile_result,
    run_result=run_result,
    comparison=comparison,
)

if comparison and not comparison.passed:
    print("mismatch:", comparison.reason)
```

Run on a board by passing a `RemoteTarget`:

```python
pipe = ModelPipeline(config, RemoteTarget(address="root@10.46.130.17"))
```

Reconstruct a config from a saved manifest, or layer config files, without the CLI:

```python
from torq.lab.types import PipelineConfig, load_config

# From a previous run's manifest:
config = PipelineConfig.from_file("model-run/manifest.json")

# Or deep-merge JSON layers (later wins):
merged = load_config(["base.json", "board.json"])
config = PipelineConfig.from_dict(merged)
```

Compare arrays directly, outside a pipeline:

```python
from torq.lab.compare import compare_outputs

result = compare_outputs(observed_arrays, expected_arrays)
print(result.passed, result.reason)
```

### Key modules

| Module                    | Responsibility                                                     |
|---------------------------|--------------------------------------------------------------------|
| `torq.lab.types`          | `PipelineConfig`, `RemoteTarget`, `CompileResult`, `RunResult`, `LabError`, `load_config`. |
| `torq.lab.pipeline`       | `ModelPipeline` — the compile / run / compare orchestrator.        |
| `torq.lab.io`             | dtype mapping, MLIR IO-spec parsing, input/output materialization. |
| `torq.lab.tools`          | discovery of the `torq-compile` / `torq-run-module` binaries.      |
| `torq.lab.compare`        | pure numeric output comparison (`compare_outputs`, `ComparisonResult`). |
| `torq.lab.remote`         | SSH/ADB staging and remote execution.                              |
| `torq.lab.manifest`       | build and atomically write `manifest.json`.                        |
| `torq.lab.profiling`      | host-profile annotation and Perfetto trace helpers (`[profile]` extra). |
