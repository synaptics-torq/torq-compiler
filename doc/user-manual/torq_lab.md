# torq.lab

`torq.lab` is a generic compile / run / verify / profile orchestration layer around `torq-compile` and `torq-run-module`. It wraps model import (ONNX/TFLite), the artifact layout, input/output handling, local **and** remote (SSH/ADB) execution, output verification, and profiling into a single reusable pipeline. It also exposes `torq-gen-config`'s per-layer executor discovery and static quantization as subcommands.

It is available via three interfaces:

- the **`torq-lab` command line**: compile, run, verify, and profile a model from the shell;
- **interactive mode**: a guided, no-flags-needed session for exploring a model; and
- the **`torq.lab` Python package**: a python module you can import into your own scripts and applications.

```{note}
`torq.lab` is bundled in the `torq-compiler` wheel (release 2.1.0 and above), so it is available wherever `torq-compile` / `torq-run-module` are.
```

---

## Installation

`torq.lab` is installed as part of the `torq-compiler` wheel; see [Quickstart → Python Wheel](getting_started.md). The `torq-lab` console script and `python -m torq.lab` entry point are available immediately after install.

Some optional features pull in extra dependencies:

```bash
# ONNX import/self-verify/gen_config/quantize
$ pip install "torq-compiler[onnx]"

# TFLite import
$ pip install "torq-compiler[tflite]"

# TFLite dynamic-to-static shape conversion (`torq-convert-static`)
# Adds the `tensorflow` dependency for TFLite flatbuffer schema
$ pip install "torq-compiler[tf]"

# Profiling annotation + Perfetto trace rendering
# Adds `pandas`, `XlsxWriter`, and `protobuf` dependencies
$ pip install "torq-compiler[profile]"

# Everything (all of the extras above)
$ pip install "torq-compiler[all]"
```

Without an extra, the features that need it raise a clear error naming the missing package.

---

## Command line

```bash
$ torq-lab <command> <model> [options]
# equivalently:
$ python -m torq.lab <command> <model> [options]
# with no command at all, torq-lab launches interactive mode (see below):
$ torq-lab
```

`<model>` is a `.onnx`, `.tflite`, or `.mlir` source for `compile`/`run`/`verify`/`profile` (imported/compiled first as needed), or a `.vmfb` module to run/verify/profile/inspect directly. It is optional when `--config` supplies `model_path`. The delegated commands (`gen_config`, `quantize`, `analyze`, `convert_dtype`, `convert_static`) do not take a `<model>` positional; see [Delegated commands](#delegated-commands).

### Subcommands

| Command      | Purpose                                                             |
|--------------|----------------------------------------------------------------------|
| `compile`    | Compile a model to a VMFB.                                          |
| `run`        | Compile if needed, then run the model.                              |
| `verify`     | Compile/run if needed, then check outputs against a reference.      |
| `profile`    | Compile/run if needed with profiling on, and report the profile.    |
| `inspect`    | Describe a `.vmfb`/`.mlir`/`.onnx`/artifact directory; suggest next commands. |
| `gen_config` | Per-layer NSS/CSS/Host executor discovery (`discover`/`run`/`view`/`edit`) and FP32→int8 QDQ quantization (`quantize`); delegates to `torq-gen-config`. |
| `quantize`   | ONNX quantization: `static` (int8 with calibration), `dynamic` (int8), or `weights` (int4/int8/bf16); delegates to `torq-quantize-model`. |
| `analyze`    | Quantization sensitivity analysis: `static`/`dynamic` (per-node) or `weights` (per-layer); delegates to `torq-quantize-model analyze`. |
| `convert_dtype` | Convert an ONNX model to Torq-compatible dtypes (`onnx`); delegates to `torq-convert-dtype`. |
| `convert_static` | Convert a dynamic TFLite model to static shapes (`tflite`); delegates to `torq-convert-static`. |
| *(none)*     | Launches [interactive mode](#interactive-mode).                     |

### Examples

Download the MobileNetV2 INT8 MLIR model from the [Synaptics Hugging Face repository](https://huggingface.co/Synaptics/MobileNetV2):

```bash
$ curl -L https://huggingface.co/Synaptics/MobileNetV2/resolve/main/MobileNetV2_int8.mlir?download=true -o MobileNetV2_int8.mlir
```

Compile a model to a VMFB (accepts `.onnx`, `.tflite`, or `.mlir`):

```bash
$ torq-lab compile MobileNetV2_int8.mlir -o mobilenetv2_int8.vmfb
$ torq-lab compile model.mlir -o model.vmfb
```

Compile and run in one step, with random inputs. `run` accepts `.onnx`/`.tflite`/`.mlir` directly and compiles automatically:

```bash
$ torq-lab run MobileNetV2_int8.mlir --random-inputs
```

Run a pre-compiled module with explicit inputs:

```bash
$ torq-lab run model.vmfb --input-npy in0.npy --input-npy in1.npy
```

Run a pre-compiled module with random inputs. `run` reads the sibling `.mlir` file for the entry function and I/O shapes. If the VMFB was compiled with `--torq-convert-dtypes --torq-convert-io-dtype`, declare that with `--convert-io-dtypes all` so the inputs and output layout use the VMFB's converted public dtypes:

```bash
$ torq-lab run model.vmfb --random-inputs --convert-io-dtypes all
```

Verify outputs against explicit goldens:

```bash
$ torq-lab verify model.mlir --random-inputs --golden golden_0.npy
```

Verify an ONNX model against a reference generated from the ONNX model itself:

```bash
$ torq-lab verify model.onnx --random-inputs
```

Profile a run:

```bash
$ torq-lab profile model.mlir --random-inputs
```

Describe an artifact and see suggested next commands (without compiling or running):

```bash
$ torq-lab inspect model.vmfb
```

### Choosing where the model runs

`--runtime-hw-type` selects the execution target. It also steers the compile: the pipeline adds the matching `torq-compile` host flags for you, so the same command line works for the simulator and for a board.

| `--runtime-hw-type` | Runs on                                   | Compile flags added                                       |
|---------------------|-------------------------------------------|-----------------------------------------------------------|
| `sim` (default)     | the cmodel and mpact simulation           | `--torq-target-host-triple=native`                        |
| `astra_machina`     | an Astra Machina board                    | none          |

A cmodel run is therefore the zero-configuration default:

```bash
$ torq-lab run model.mlir --random-inputs
```

Running on a board additionally needs an address for the pipeline to stage artifacts and execute remotely:

```bash
$ torq-lab run model.mlir --random-inputs \
    --runtime-hw-type astra_machina \
    --remote <board IP/ADB address>
```

See [Remote execution](#remote-execution) for the SSH and ADB address forms.

### Options

Running `<command> -h` displays the options relevant to the command.

**common**: `--config`, `--work-dir`, `--timeout`, `--reuse-work-dir`

| Option                  | Description                                                        |
|-------------------------|----------------------------------------------------------------------|
| `--config PATH`         | Config JSON file(s) to layer (repeatable, later wins). An alternative to the flags below; see [Config files and layering](#config-files-and-layering). |
| `--work-dir DIR`        | Artifact directory (default: `./<model-stem>-run`).                |
| `--timeout SECONDS`     | Per-tool timeout in seconds; see note below.                  |
| `--reuse-work-dir`      | Reuse a populated work directory instead of allocating a new run ID. |

**target** (`compile`, `run`, `verify`, `profile`): `--runtime-hw-type`

**compile stage** (`compile`, `run`, `verify`, `profile`):

| Option                     | Description                                                        |
|----------------------------|--------------------------------------------------------------------|
| `--chip CHIP`              | Chip target for `--torq-hw` (default: `SL2610`).                  |
| `--compiler-option OPT`    | Extra `torq-compile` flag (repeatable).                           |
| `--dump-ir`                | Dump IR after each pass into `<work-dir>/debug/ir`.                |
| `--dump-phases`            | Dump compilation phases into `<work-dir>/phases`.                 |
| `--profile-compile`        | Enable compile-time profiling.                                    |

**run stage** (`run`, `verify`, `profile`):

| Option                     | Description                                                        |
|----------------------------|--------------------------------------------------------------------|
| `--function NAME`          | Entry function name. Rarely needed as it is parsed from the model automatically (falling back to `main`); set it only to pick a specific function out of a multi-function module. |
| `--runtime-option OPT`     | Extra `torq-run-module` flag (repeatable).                        |
| `--convert-io-dtypes ...`  | Declare converted VMFB public I/O: `all`, selected `input:IDX` / `output:IDX`, or all except `!input:IDX` / `!output:IDX`. Must match the VMFB's compilation. |
| `--input-npy PATH`         | Input `.npy` file (repeatable).                                   |
| `--input-spec SPEC`        | Input tensor spec, e.g. `1x1x64xbf16` (repeatable; for a VMFB with no source. See [Running a VMFB with no source](#running-a-vmfb-with-no-source)). |
| `--output-spec SPEC`       | Output tensor spec, same form as `--input-spec` (repeatable).      |
| `--seed N`                 | Random seed for input generation (default: `1234`).               |
| `--random-inputs`          | Generate random inputs from the model's I/O spec.                 |

**verification** (`verify` only):

| Option                  | Description                                                        |
|-------------------------|----------------------------------------------------------------------|
| `--golden PATH [PATH ...]` | Expected output `.npy` file(s) to compare against (repeatable).   |
| `--reference PROVIDER`  | Golden reference provider: `onnx:PATH`, or `onnx` to use the model's own ONNX source. Defaults to the model itself when it is `.onnx` and no `--golden` is given. See [Self-verification](#self-verification). |

**remote target** (`run`, `verify`, `profile`):

| Option                 | Description                                                  |
|------------------------|--------------------------------------------------------------|
| `--remote ADDR`        | Remote board address (see forms in [Remote execution](#remote-execution)). |
| `--remote-port PORT`   | SSH port (default: `22`).                                    |
| `--remote-key PATH`    | SSH private key path.                                        |
| `--remote-runner PATH` | Absolute path of `torq-run-module` already on the device.   |
| `--stage-runner PATH`  | Local `torq-run-module` to copy onto the device.             |

**output** (`compile`, `run`, `verify`, `profile`, `inspect`):

| Option           | Description                                                                 |
|------------------|-------------------------------------------------------------------------------|
| `--print-plan`   | Print what the command would do and exit, without importing, compiling, running, or connecting to a board. |
| `--json`         | Print the result as one structured JSON object instead of human-readable text. |

`compile` also has `-o, --output PATH` (output VMFB path; default `<work-dir>/<model-stem>.vmfb`). `inspect` also has `--artifact MANIFEST` (resolve facts from an explicit manifest), `--source MODEL` (resolve I/O from an explicit `.mlir`/`.onnx`), and `--function NAME` (override the resolved entry function).

`--timeout` is a single value applied *independently* to each tool invocation, not a budget for the whole pipeline: `run` gives the compile its own `SECONDS` and the run another `SECONDS`. There is currently no way to set a different limit for the compiler and the runtime. On a remote run the same value also bounds each individual command issued over the SSH/ADB transport (default `15` when `--timeout` is unset).

When both `--config` and CLI flags are given, an explicit CLI flag always overrides the same key from the config file.

### Exit codes

| Code | Meaning                                                             |
|------|-----------------------------------------------------------------------|
| `0`  | Success (indicates outputs matched for `verify`).                       |
| `1`  | Error: a tool failed, or `verify` had no reference to check against. |
| `2`  | Verification failed: the run's outputs did not match the reference. |

A run that produced no outputs (or a mismatched output count) while a reference was configured is reported as a mismatch (exit `2`), not a pass.

Every invocation writes a `manifest.json` into the work dir (see [the work directory and manifest](#the-work-directory-and-manifest)).

---

## Model import

`compile`, `run`, `verify`, `profile`, and interactive mode all accept `.onnx` and `.tflite` sources directly alongside `.mlir` and the model is imported to `<work-dir>/<model-stem>.mlir` automatically before compiling:

- `.onnx` is imported via `torq.lab.model_tools.importers.onnx.convert_onnx_to_mlir`.
- `.tflite` is imported via `torq.lab.model_tools.importers.tflite.convert_tflite_to_mlir`, which delegates to `tosa-converter-for-tflite`.
- `.mlir` is used unchanged.
- `.vmfb` is run directly; it is never a valid input to `compile`.

Both import paths are lazy: compiling or running a `.mlir` model never pulls in the `onnx` or `tflite` extras.

`inspect` additionally accepts an artifact directory (a `--work-dir` from a previous run).

---

## Self-verification

`verify` compiles/runs the model, then checks the outputs against a reference. It needs one of:

- `--golden PATH [PATH ...]` — explicit expected-output `.npy` files; or
- `--reference onnx:PATH` (or plain `--reference onnx` to use the model's own ONNX source).

Verifying an `.onnx` model needs neither flag: `torq-lab verify model.onnx` defaults to the model itself as the reference. Verifying a `.mlir` or `.vmfb` model needs an explicit `--golden` or `--reference onnx:PATH`, since there is no ONNX source to fall back to.

```{note}
A `--golden` value that looks like a model file (`.onnx`/`.tflite`/`.mlir`/`.vmfb`) placed *before* the model positional can be swallowed by `--golden`'s repeatable parsing. If `verify` reports `'model' is required`, check the flag order; the error message includes a corrected command line.
```

When a reference is generated from ONNX (rather than supplied as goldens), it is computed in two tiers: first natively with `onnxruntime`, falling back to a numpy execution path (handling bf16 `MatMul`/`Einsum`/pooling/`Gelu`, which `onnxruntime` may lack kernels for) if that fails. If neither tier can execute the model, `verify` fails with an error naming the op it could not run and pointing you at `--golden`.

---

## Provenance and inspection

`torq-lab inspect MODEL` describes an artifact: its resolved entry function, I/O signature, chip, debug info, and compile command without compiling or running the model. It prints copy-pasteable `run`/`verify`/`profile` hints and, for a `.vmfb`, whether an annotated profile is possible.

```bash
$ torq-lab inspect model-run/model.vmfb
```

Each resolved fact is tagged with where it came from. Facts are resolved in this order, the first available source winning:

1. An explicit manifest (`--artifact MANIFEST`).
2. A `manifest.json` colocated with the model.
3. A sibling `.mlir`/`.onnx` next to a `.vmfb` (or an explicit `--source`).
4. The I/O spec parsed from that MLIR source.
5. Explicit `--input-spec`/`--output-spec` values.

If the entry function still cannot be resolved after all of the above, `inspect` falls back to `"main"` and marks it as a guess.

`--print-plan` (on `compile`/`run`/`verify`/`profile`/`inspect`) exits after printing what the command would do: the model, any import/compile output, the execution target, and the input source. `--json` prints the same information as one JSON object instead of human-readable text.

On a remote run, a transport failure is reported by named stage: `connect`, `stage-runner`, `stage-model`, `execute`, or `pull-results`.

---

## Delegated commands

Five `torq-lab` commands are thin delegations to standalone tools installed in the compiler wheel. Unlike `compile`/`run`/`verify`/`profile`/`inspect`, they do not take a `<model>` positional: each keeps the option syntax of the tool it wraps (run `torq-lab <command> --help` for the full reference).

| `torq-lab` command  | Delegates to          | Subcommands                                | Purpose                                            |
|---------------------|-----------------------|--------------------------------------------|----------------------------------------------------|
| `gen_config`        | `torq-gen-config`     | `discover`, `run`, `view`, `edit`, `quantize` | Per-layer NSS/CSS/Host executor discovery; FP32→int8 QDQ quantization. |
| `quantize`          | `torq-quantize-model` | `static`, `dynamic`, `weights`, `analyze`  | ONNX quantization.                                 |
| `analyze`           | `torq-quantize-model` | `static`, `dynamic`, `weights`             | Quantization sensitivity analysis.                 |
| `convert_dtype`     | `torq-convert-dtype`  | `onnx`                                     | Convert an ONNX model to Torq-compatible dtypes.   |
| `convert_static`    | `torq-convert-static` | `tflite`                                   | Convert a dynamic TFLite model to static shapes.   |

### gen_config

`torq-lab gen_config <discover|run|view|edit|quantize>` — per-layer executor discovery. See [torq-gen-config](torq-gen-config.md) for the full command reference; every option works the same way under either form. `gen_config quantize` quantizes an FP32 ONNX model to integer QDQ with calibration.

```bash
$ torq-lab gen_config discover --model model.onnx --output-dir results/ --skip-mode
$ torq-lab gen_config quantize --model model.onnx --output model.int8.onnx
```

### quantize

`torq-lab quantize <static|dynamic|weights>` — ONNX quantization. All three take `-i, --input` (FP32 ONNX model, required) and `-o, --output` (quantized ONNX model).

**`static`** — int8 quantization with calibration. `--output` defaults to `<input-stem>.int8.onnx`.

| Option | Description |
|--------|-------------|
| `--num-calib N` | Number of synthetic calibration samples (default: 20). |
| `--per-channel` | Use per-channel weight quantization. |
| `--full-integer` | Rewrite I/O to integer (remove input Q and output DQ nodes). |
| `--quant-format {qdq,qoperator,hybrid}` | ONNX quantization format (default: `qdq`); `hybrid` picks `qoperator` for ops with good `qoperator` support (Conv, Add, MatMul, ...) and `qdq` for the rest, applied per layer. |
| `--quant-dtype {A8W8}` | Quantized activation/weight dtype combination (currently only `A8W8`). |
| `--quantize-only-ops OPS ...` | Only quantize the given ONNX op types. |
| `--quantize-only-nodes NODES ...` | Only quantize the given node names. |
| `--exclude-nodes NODES ...` | Exclude the given nodes from quantization (e.g. an `analyze` exclude list). |

**`dynamic`** — int8 quantization via onnxruntime (no calibration).

| Option | Description |
|--------|-------------|
| `--quantize-only-ops OPS ...` | Only quantize the given ONNX op types. |
| `--quantize-only-nodes NODES ...` | Only quantize the given node names. |
| `--exclude-nodes NODES ...` | Exclude the given nodes from quantization (e.g. an `analyze` exclude list). |
| `--skip-preprocess` | Skip onnxruntime pre-processing steps that may improve quantization quality. |
| `--uint8-weights` | Generate unsigned integer weights. |
| `--per-tensor` | Quantize weights per tensor instead of per channel. |

**`weights`** — weight-only int4/int8/bf16 quantization of `MatMul` weights (LLM-oriented). Requires `--bits` or `--config`.

| Option | Description |
|--------|-------------|
| `--bits {4,8,16}` | Uniform bit-width (4=int4, 8=int8, 16=bf16); ignored with `--config`. |
| `--block-size N` | Block size for block quantization (default: 32). |
| `--config PATH` | Per-layer quantization config JSON from `analyze weights` (overrides `--bits` for mixed quantization). |
| `--dequantize-weights` | Dequantize the weights and emit a single bf16 model ready for IREE compilation (no DQL nodes). |
| `--skip-layers SUBSTR ...` | Layer-name substrings to skip (e.g. `lm_head`). |

```bash
$ torq-lab quantize static -i model.onnx -o model_static.onnx
$ torq-lab quantize dynamic -i model.onnx -o model_dynamic.onnx
$ torq-lab quantize weights -i model.onnx -o model_weights.onnx --bits 8
```

### analyze

`torq-lab analyze <static|dynamic|weights>` — quantization sensitivity analysis (also reachable as `torq-lab quantize analyze <method>`). All three take `-i, --input` (FP32 ONNX model) and `-o, --output` (sensitivity report JSON). The reports feed back into `quantize`: `static`/`dynamic` can write an exclude list for `--exclude-nodes`, and `weights` can write the per-layer config for `--config`.

**`static`** and **`dynamic`** rank nodes per node and share these options:

| Option | Description |
|--------|-------------|
| `--exclude-output PATH` | Also write nodes at/above `--exclude-class` as a JSON list usable with `quantize <method> --exclude-nodes`. |
| `--exclude-class {MEDIUM,HIGH,CRITICAL}` | Severity at/above which a node joins the exclude list (default: `HIGH`). |
| `--op-types OPS ...` | Node op types to test (default: `MatMul Gemm`). |
| `--skip-nodes SUBSTR ...` | Node-name substrings to skip. |
| `--calibration-data NPZ` | `.npz` of input feeds (keys = model input names); seeded random inputs when omitted. |
| `--seed N` | Seed for random calibration inputs (default: 42). |

`static` additionally takes `--num-calib N` (default: 1) and the quantization format flags of `quantize static` (`--per-channel`, `--quant-format`, `--quant-dtype`); `dynamic` takes `--uint8-weights`, `--per-tensor`, and `--skip-preprocess`.

**`weights`** ranks `MatMul` layers per layer for LLM-style weight quantization. `--embeddings NPY` (token embedding table) is required; other notable options: `--tokenizer JSON` (prompt tokenization), `--bits ...` (bit-widths to test, default: `4 8 16`), `--config-output PATH` (per-layer quantization config JSON for `quantize weights --config`), prompt options (`--prompts`, `--prompts-file`, `--pre-tokenized-file`, `--chat-template`, `--system-prompt`, `--num-tokens`), and the KL-divergence thresholds `--bf16-threshold` (default: 0.1) and `--int8-threshold` (default: 0.01).

```bash
$ torq-lab analyze dynamic -i model.onnx -o report.json
$ torq-lab analyze weights -i model.onnx -o report.json --embeddings token_embeddings.npy
```

### convert_dtype

`torq-lab convert_dtype onnx` — convert an ONNX model to Torq-compatible dtypes (the same command is available standalone as `torq-convert-dtype onnx`).

| Option | Description |
|--------|-------------|
| `-i, --input` | Input ONNX model path (required). |
| `-o, --output` | Output ONNX model path (required). |
| `-d, --dtype` | Export dtype: `bf16`, `fp16`, `int32`, `int16`, or `int8` (required). |
| `--opset N` | ONNX opset to use; a relatively new opset is required for bf16 support in some ops (default: 22). |
| `--max-float X` | Maximum FP32 magnitude for constant conversion; initializers above it or the export dtype's range are rejected to avoid an overflowing conversion (default: 1e9). |
| `--convert-io` | Convert model I/O to the export dtype. |
| `--modelopt` | Use TensorRT modelopt for dtype conversion. |
| `--bf16-rounding {nearest,truncate}` | Rounding mode for fp32→bf16 constants (default: `nearest`); `truncate` drops the low 16 mantissa bits. |
| `--enforce-io-casts` | Insert Cast nodes so ONNX-spec-mandated int64 operator I/O (Reshape shape, Slice params, Shape/Size outputs) stays int64. |
| `--strip-unused-outputs` | Strip unused node outputs during post-conversion cleanup; may break fixed-output nodes like TopK. |
| `--torq-onnx-finalize` | Run Torq-oriented ONNX post-processing (ORT symbolic shapes, IR cap, value_info cleanup). |

```bash
$ torq-lab convert_dtype onnx -i model.onnx -o model_bf16.onnx -d bf16 --convert-io
```

### convert_static

`torq-lab convert_static tflite` — convert a dynamic TFLite model to static shapes, using the default shapes (the same command is available standalone as `torq-convert-static tflite`). Requires the `[tf]` extra for the TFLite flatbuffer schema.

| Option | Description |
|--------|-------------|
| `-i, --input` | Input TFLite model path (required). |
| `-o, --output` | Output TFLite model path (required). |

```bash
$ torq-lab convert_static tflite -i model.tflite -o model_static.tflite
```

---

## Config files and layering

Instead of a long flag list you can supply one or more JSON config files with `--config`. Files are deep-merged in order (later wins), docker-compose style: keep a shared base and overlay per-environment differences. Any explicit CLI flag overrides the same key from the config file.

The same schema drives both `torq-lab` and `torq-gen-config`: `PipelineConfig` keys at the top level (`model_path`, `chip`, `runtime_hw_type`, `timeout`, `compiler_options`, ...), a `remote` section shared by both tools, and a `gen_config` section for discovery-only options (`skip_mode`, `collect_timing`, ...) that `torq-lab` ignores and `torq-gen-config` reads.

```json
// base.json
{
  "model_path": "model.onnx",
  "chip": "SL2610",
  "compiler_options": ["--some-flag"],
  "random_inputs": true,
  "gen_config": {
    "skip_mode": true,
    "collect_timing": true
  }
}
```

```json
// board.json: overlay adding a remote target
{
  "runtime_hw_type": "astra_machina",
  "remote": { "address": "root@10.46.130.17" }
}
```

`remote.address` takes the same forms as `--remote`, so an ADB-attached board is configured the same way: `"adb"` for the first attached device, or an explicit serial:

```json
// board-adb.json: the same board over ADB instead of SSH
{
  "runtime_hw_type": "astra_machina",
  "remote": { "address": "adb", "stage_runner": "./torq-run-module" }
}
```

The same file drives both tools: `run` uses the top-level keys and `remote`, `gen_config discover` uses the top-level keys, `remote`, and `gen_config`:

```bash
$ torq-lab run --config base.json --config board.json
$ torq-lab gen_config discover --config base.json
```

Note that the config files you pass in and the `manifest.json` a run writes out are different directions of the same schema, not competing formats: `base.json` and `board.json` are **inputs** you author, while `manifest.json` is an **output** the pipeline records (see [The work directory and manifest](#the-work-directory-and-manifest)). Because a manifest embeds the config it ran with, it can be fed straight back in as a config in a new run.

Explicit positional/flag values still override the layered files (e.g. a positional `<model>` overrides `model_path`; `--remote` overrides the config's `remote.address`). Each source may be a bare config dict **or** a full `manifest.json`.

---

## Remote execution

`torq.lab` can stage the module and inputs onto a board, run `torq-run-module` there, and pull the outputs back. Enable it with `--remote`.

```bash
$ torq-lab run model.mlir \
    --random-inputs \
    --runtime-hw-type astra_machina \
    --remote root@10.46.130.17 \
    --stage-runner ./torq-run-module
```

`--remote` accepts the same address forms as the transport factory:

- `adb` — the first attached ADB device;
- an ADB serial number;
- `user@host` or a bare hostname/IP over SSH.

See [Options](#options) for the remote flags. A transport failure names the stage it happened at, see [Provenance and inspection](#provenance-and-inspection).

---

## Profiling

Passing `--profile-compile` turns on compile-time profiling; the dedicated `profile` command turns on runtime profiling. The compile always emits debug info, and (with the `[profile]` extra installed) the pipeline adds the relevant traces and renders a Perfetto viewer into the profiles directory.

**Compile-time profiling** (`--profile-compile`, on `compile`/`run`/`verify`/`profile`) needs only compilation. `torq-lab compile --profile-compile` writes `compile_profile.csv`, the compile Perfetto trace(s) (`*_compile.pb`), and `perfetto_viewer.html`:

```bash
$ torq-lab compile model.mlir --profile-compile
```

**Runtime profiling** uses the `profile` command, which compiles (with profiling enabled, if a source needs compiling) and runs, then annotates the runtime host profile against the compile-time debug info and renders the viewer:

```bash
$ torq-lab profile model.mlir --random-inputs
```

By default each `profile` run lands in its own timestamped subdirectory (`<work-dir>/profiles/<UTC-timestamp>`) so repeated measurements never overwrite each other. Combine `--profile-compile` with `profile` to merge the compile and runtime traces in one viewer.

`profile` classifies what it actually produced:

| Quality     | Meaning                                                                 |
|-------------|--------------------------------------------------------------------------|
| `annotated` | A full annotated report and/or Perfetto viewer was produced.            |
| `raw`       | Only the host profile CSV was produced (no debug info to annotate against, or the `[profile]` extra is not installed). |
| *(neither)* | Profiling produced nothing usable; `profile` exits `1`.                 |

For the meaning of the trace columns and the annotated-profile format, see the [Performance Profiling Tool](profiling.md) chapter.

---

## Running a VMFB with no source

A `.vmfb` delivered on its own (no sibling `.mlir`/`.onnx`, no `manifest.json`) has no recorded I/O signature. Supply one explicitly with repeatable `--input-spec`/`--output-spec` (tensor-type strings like `1x1x64xbf16`):

```bash
$ torq-lab run model.vmfb --input-spec 1x1x64xbf16 --output-spec 1x1x1000xf32 --random-inputs --seed 42
```

`--seed` makes random-input generation deterministic across runs. If no input signature is available at all, the command fails with recovery suggestions.

By default, running into a work directory that already has `inputs/`/`outputs/` from a previous run allocates a new run ID (an 8-character suffix on the work-dir name) so the two runs' artifacts don't mix. Pass `--reuse-work-dir` to run back into the same directory instead.

---

## Interactive mode

Running `torq-lab` (or `python -m torq.lab`) with no arguments launches an interactive session: it prompts for a model, compiles it, then loops a menu of actions over it.

```
$ torq-lab
torq-lab interactive mode: press Ctrl-D to quit at any time.
Model file (.onnx/.tflite/.mlir/.vmfb): model.onnx
Work dir [/home/user/model-run]:
Chip [SL2610]:
Runtime HW type [sim]:
Extra --compiler-option flags [none]:
Remote address [none = local]:
Importing model.onnx -> MLIR ...
Compiling for SL2610 (target: sim)... this can take a while for larger models.
Compile finished.
Compiled: /home/user/model-run/model.vmfb
Function: main
Inputs: ['1x3x224x224xf32']
Outputs: ['1x1000xf32']
Next: torq-lab run /home/user/model-run/model.vmfb

Menu:
  1) run
  2) profile
  3) verify
  4) inspect
  5) gen_config
  6) quantize
  7) recompile
  8) quit
Choice: 1
Inputs: 1) random  2) npy paths [1]: 1
Remote address [local]:
Preparing the model (compiling if needed)...
Running on sim...
Run finished.
run: ok
  run: 0.42s
  target: sim
  inputs: random (seed=1234)
  next: torq-lab inspect /home/user/model-run/model.vmfb

Menu:
  1) run
  2) profile
  3) verify
  4) inspect
  5) gen_config
  6) quantize
  7) recompile
  8) quit
Choice: quit
```

A `.vmfb` model skips the compile-flags prompts entirely. `recompile` re-prompts the compile flags and rebuilds in place; it is a no-op for a `.vmfb`. `gen_config` prompts for the action (`discover`/`run`/`view`/`edit`) and forwards to `torq-gen-config`; `quantize` prompts for the output path, `--num-calib`, `--per-channel`, and `--full-integer`, runs `torq-gen-config quantize` (FP32→int8 QDQ), and can optionally adopt its int8 output as the new model and recompile it. `Ctrl-D` (or `Ctrl-C`) exits cleanly at any prompt.

---

## The work directory and manifest

Everything a run produces lands under `--work-dir` (default `./<model-stem>-run`):

```
<work-dir>/
├── model.vmfb          # compiled module
├── manifest.json       # config + results + artifact record (see below)
├── inputs/             # materialized .bin / .npy inputs
├── outputs/            # raw output_<i>.bin files
├── debug/              # --torq-debug-info (and ir/ under --dump-ir)
├── phases/             # --dump-phases
├── profiles/           # host_profile.csv, traces, perfetto_viewer.html
└── remote/             # scratch for staged remote runs
```

`manifest.json` keeps three concerns separate:

- **`config`**: the pipeline inputs, round-trippable back into a `PipelineConfig` (and a `RemoteTarget`, when present). This lets a run be reconstructed from its manifest.
- **`results`**: what the run produced: the `compile` / `run` commands and timings, output paths, profiling artifacts, the `comparison` verdict, and any `diagnostics`.
- **`artifact`**: the resolved entry function, entry points, I/O spec, converted-I/O state, debug dir, chip, compile command, and source. This is what `torq-lab inspect` reads back without recompiling.

It is written atomically, so a concurrent reader always sees a complete file.

---

## Python library

The same pipeline is importable. The typical flow builds a `PipelineConfig`, drives a `ModelPipeline`, and inspects the structured results:

```python
from pathlib import Path

from torq.lab.pipeline.remote import RemoteTarget
from torq.lab.pipeline.workflow import ModelPipeline, PipelineConfig

config = PipelineConfig(
    model_path=Path("model.onnx"),
    work_dir=Path("model-run"),
    chip="SL2610",
    random_inputs=True,
    reference="onnx",
)

pipe = ModelPipeline(config)
compile_result, vmfb_path = pipe.ensure_vmfb()  # imports + compiles if needed
run_result = pipe.run(vmfb_path)
comparison = pipe.verify_outputs(run_result)     # goldens, else a generated reference, else None
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

Profile a run:

```python
compile_result, run_result = pipe.profile()
```

Describe an artifact without a `ModelPipeline`:

```python
from torq.lab.pipeline import artifacts

info = artifacts.describe("model-run/model.vmfb")
print(info.function, info.io_spec, info.provenance)
```

Reconstruct a config from a saved manifest, or layer config files, without the CLI:

```python
from torq.lab.pipeline.workflow import PipelineConfig, load_config

# From a previous run's manifest:
config = PipelineConfig.from_file("model-run/manifest.json")

# Or deep-merge JSON layers (later wins):
merged = load_config(["base.json", "board.json"])
config = PipelineConfig.from_dict(merged)
```

Compare arrays directly, outside a pipeline:

```python
from torq.lab.verification.compare import compare_outputs

result = compare_outputs(observed_arrays, expected_arrays)
print(result.passed, result.reason)
```

Or generate a reference from an ONNX model directly:

```python
from torq.lab.verification.reference import onnx_reference_outputs

expected = onnx_reference_outputs("model.onnx", input_arrays)
```

### Key modules

| Module                    | Responsibility                                                     |
|---------------------------|--------------------------------------------------------------------|
| `torq.lab`                | `LabError` (base error) and `Case` (named case container) shared across functional owners. |
| `torq.lab.pipeline.workflow` | `PipelineConfig`, `CompileResult`, `RunResult`, `load_config`, and `ModelPipeline`: the import / compile / run / verify / profile orchestrator. |
| `torq.lab.pipeline.io`        | MLIR IO-spec parsing, dtype mapping, input/output materialization, and the I/O dtype-conversion policy. |
| `torq.lab.pipeline.artifacts` | resolve a VMFB/model's metadata and provenance (`describe`, `ArtifactInfo`) and build/atomically write `manifest.json`. |
| `torq.lab.pipeline.remote`    | `RemoteTarget`, SSH/ADB transport, and remote staging/execution (`RemoteExecutor`). |
| `torq.lab.pipeline.tools`     | discovery of the `torq-compile` / `torq-run-module` binaries.      |
| `torq.lab.verification.compare`  | pure numeric output comparison (`compare_outputs`, `ComparisonResult`). |
| `torq.lab.verification.reference` | ONNX/numpy reference implementations (`onnx_reference_outputs` and friends) used to generate a verification golden. |
| `torq.lab.cli.output`     | `Plan`/`Summary` dataclasses and their human/JSON formatters (`format_plan`, `format_summary`, `format_inspect`). |
| `torq.lab.cli.interactive` | the no-argument interactive session (`run_interactive`).           |
| `torq.lab.profiling`      | host-profile annotation and Perfetto trace helpers (`[profile]` extra). |
