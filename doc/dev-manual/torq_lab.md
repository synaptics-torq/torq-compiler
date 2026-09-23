# torq.lab developer guide

`torq.lab` is the generic compile / run / verify / profile orchestration layer around `torq-compile` and `torq-run-module`. It is exposed three ways: the `torq-lab` command line, a no-argument interactive session, and an importable Python package. End-user usage is documented in the [torq.lab user guide](https://synaptics-torq.github.io/torq-compiler/v/latest/user-manual/torq_lab.html); this page is for people modifying the code.

## Source layout

| Path | Contents |
|------|----------|
| `cli/` | The `torq-lab` command line: `parser.py` (subcommands + option groups), `commands.py` (dispatch, `main()`, delegated-command interception), `interactive.py` (the no-argument session), `output.py` (`Plan`/`Summary` dataclasses and their human/JSON formatters). |
| `pipeline/` | `workflow.py` (`PipelineConfig`, `ModelPipeline`, result types, config-file layering), `io.py` (MLIR I/O-spec parsing, input/output materialization, the converted-dtype policy), `artifacts.py` (`describe`, `manifest.json`), `remote.py` (`RemoteTarget`, SSH/ADB transports, remote staging/execution), `tools.py` (discovery of the `torq-compile` / `torq-run-module` binaries). |
| `verification/` | `compare.py` (numeric output comparison) and `reference.py` (ONNX/numpy golden-reference generation). |
| `quantization/onnx/` | The `static` / `dynamic` / `weights` quantization and sensitivity-analysis implementations; `cli.py` is the `torq-quantize-model` entry point that `torq-lab quantize` and `torq-lab analyze` delegate to. |
| `model_tools/` | `importers/` (ONNX, TFLite), `dtype_conversion/` (`torq-convert-dtype`), `shape_conversion/` (`torq-convert-static`), `extraction/` (layer/subgraph extraction). |
| `profiling/` | Host-profile annotation and Perfetto trace rendering (the `[profile]` extra; the module must import cleanly without it). |
| `utils/`, `logging.py`, `metrics.py`, `reporting.py` | Shared helpers. |

`torq/lab/__init__.py` defines the package-wide types no single functional owner takes: `LabError` (base error) and `Case` (named case container).

## Core API

Two entry points cover most day-to-day work: the pipeline classes in `torq.lab.pipeline.workflow` (which the CLI handlers call) and the CLI's `main()` (callable in-process).

### Driving the pipeline

```python
from torq.lab.pipeline.workflow import ModelPipeline, PipelineConfig

config = PipelineConfig(
    model_path="mm_add.onnx",       # .onnx/.tflite/.mlir is imported to MLIR first
    work_dir="mm_add-run",
    chip="SL2610",
    runtime_hw_type="sim",           # "astra_machina" for a board build
    input_npy=["mm_add_input.npy"],
    reference="onnx",                # verify against the model's own ONNX reference
)
pipe = ModelPipeline(config)

compile_result, run_result = pipe.compile_run()
print(compile_result.vmfb_path)      # mm_add-run/mm_add.vmfb
print(run_result.outputs[0])         # numpy array; raw .bin files in output_paths

comparison = pipe.verify_outputs(run_result)   # None if no golden/reference is configured
pipe.write_manifest(compile_result=compile_result, run_result=run_result,
                    comparison=comparison, status="ok")

# manifest.json round-trips back into a config, so the run is reproducible:
config2 = PipelineConfig.from_file("mm_add-run/manifest.json")
```

Notes for `workflow.py`:

- `compile()` and `run()` can be called separately; `ensure_vmfb()` returns a pre-compiled VMFB from the work dir instead of recompiling. `profile()` sets `profile_runtime` *before* `ensure_vmfb()` (so a source that needs compiling gets `--torq-enable-profiling` on) and returns `(compile_result, run_result)` with the `RunResult` profiling fields populated.
- `verify_outputs()` compares against explicit goldens if configured, else a generated reference (onnxruntime, falling back to the numpy path), else returns `None`. The `ComparisonResult` it returns carries `.passed` / `.reason`.
- `CompileResult.command` / `RunResult.command` record the exact `torq-compile` / `torq-run-module` argv — the in-tree tests assert on them, so keep the flag spelling stable.

### Running the CLI in-process

Every `torq-lab` command is a `main(argv) -> int` function, so the CLI can be driven from Python with the same exit codes as the shell (`0` ok, `1` error, `2` verification failed):

```python
from torq.lab.cli import commands as lab_cli

rc = lab_cli.main(["verify", "mm_add.onnx", "--work-dir", "mm_add-run",
                   "--random-inputs", "--reference", "onnx"])
```

The delegated mains have the same signature: `lab_cli.main(["quantize", "static", "-i", "model.onnx", ...])` intercepts the task name and forwards `argv[1:]` to the tool's own parser, which is also how `commands.main()` keeps the tools' heavy imports out of parse time.

## Packaging

`torq.lab` ships in the `torq-compiler` wheel:

- Optional dependencies are declared as extras in `compiler/setup.py` (`onnx`, `tflite`, `tf`, `profile`, and `all`), and the features that need them must import them lazily so the base wheel stays lean. A missing extra surfaces as a clear error naming the package.
- Console scripts (`torq-lab`, `torq-gen-config`, `torq-quantize-model`, `torq-convert-dtype`, `torq-convert-static`) are registered in `compiler/setup.py`.

## Adding a command

`torq-lab` commands come in two shapes (see `cli/commands.py`):

1. **Native pipeline subcommands** (`compile` / `run` / `verify` / `profile` / `inspect`): add the subparser in `cli/parser.py` (`_build_parser`) and a `_cmd_*` handler in `cli/commands.py`. Options flow through `PipelineConfig` (and `RemoteTarget`), which gives you config-file layering (`--config`), `--print-plan`, and `--json`. Subparsers keep `allow_abbrev=False`.
2. **Delegated standalone tools** (`gen_config` / `quantize` / `analyze` / `convert_dtype` / `convert_static`): the tool implements its own `main()` under `python/torq/...` and registers a console script in `compiler/setup.py`; `commands.main()` intercepts the task name *before* the parser is built so the tool's heavy dependencies (onnxruntime, tensorflow) are not imported at parse time. The delegated groups are listed in the parser epilog, which is how `torq-lab --help` advertises them.
