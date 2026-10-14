# Debugging compile errors

When a compile fails, `torq-lab` reports what failed, where it happened, and what to try next. `compile`, `run`, `verify`, and `profile` diagnose compile failures automatically. Interactive mode does this too.

## Start here

Run `debug` with the same model and options as the failing compile:

```bash
torq-lab debug model.onnx --work-dir model-debug
```

It accepts `.onnx`, `.tflite`, and `.mlir` files. Install `torq-lab` with the [compiler wheel](torq_lab.md#installation). You can also invoke it as `python -m torq.lab debug`.

Keep target and compiler options when reproducing a failure:

```bash
torq-lab debug model.onnx \
    --chip SL2610 \
    --work-dir model-debug \
    --compiler-option=--torq-disable-host
```

Use `--compiler-option=VALUE` with the equals sign. Repeat it for each compiler option.

The report starts with a diagnosis class. Read the `message` and `hint` first. The `op`, `source op`, and `location` identify the affected operation when available. For example, a report may contain:

```text
diagnosis: frontend-unsupported-op
  op: torch.operator
  source op: onnx.LpPool
  hint: The op is missing from the torch-mlir ONNX lowering; re-export around it or request the op.
```

The examples below show selected report lines. Pass names and locations depend on the compiler build and model.

### Try alternative compiler options

Add `--probe` to try relevant options one at a time:

```bash
torq-lab debug model.mlir --probe
```

A passing probe suggests a workaround. Copy that option into your next `compile` command, then verify the resulting model. A failed probe leaves the original diagnosis in place.

### Find failing ONNX layers

Add `--bisect` to compile each ONNX layer separately:

```bash
torq-lab debug model.onnx --bisect
```

The report lists failing layers by class. If every layer passes, investigate fusion, tiling, or memory use in the full model. Every extracted layer is checked, regardless of model size.

### Limit the time spent diagnosing

Each tool run has a default timeout of 600 seconds. Override it for a larger model:

```bash
torq-lab debug model.onnx --timeout 1800
```

Diagnosis may compile the model twice to collect reproduction files. Probes and layer checks add more runs, each with its own timeout.

To skip automatic diagnosis on a normal compile:

```bash
torq-lab compile model.onnx --no-auto-debug
```

Automatic diagnosis applies to compile failures. A failure while running an already compiled model needs separate investigation.

## Diagnosis class index

Use the class printed in your report to jump to its explanation and remedies. The classifier checks specific diagnostic signatures before broader ones. A timeout takes precedence over text; a signal without a recognized diagnostic is reported as a crash.

Classes: [environment](#environment), [assert](#assert), [fatal-error](#fatal-error), [invalid-model](#invalid-model), [import-failed](#import-failed), [dynamic-shape](#dynamic-shape), [invalid-source](#invalid-source), [frontend-unsupported-op](#frontend-unsupported-op), [kernel-unsupported](#kernel-unsupported), [no-npu-lowering](#no-npu-lowering), [serialization](#serialization), [memory](#memory), [tiling](#tiling), [input-conversion](#input-conversion), [fallback-failed](#fallback-failed), [legalization](#legalization), [non-convergence](#non-convergence), [timeout](#timeout), [silent-pass-failure](#silent-pass-failure), [crash](#crash), [unknown](#unknown).

## Common failures and fixes

### no-npu-lowering

An operation has no NPU implementation, and its CSS or Host fallback is disabled. The report names the source node and tests whether enabling a fallback lets it compile.

To try this with a small model, save the following as `make_erf.py` and run `python make_erf.py`. It requires the `onnx` package:

```python
import onnx
from onnx import TensorProto, helper

x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 64])
y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 64])
graph = helper.make_graph(
    [helper.make_node("Erf", ["x"], ["y"], name="erf")], "erf", [x], [y]
)
model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
onnx.save(model, "erf_f32.onnx")
```

Disable both fallbacks to reproduce the failure:

```bash
torq-lab debug erf_f32.onnx \
    --compiler-option=--torq-disable-css \
    --compiler-option=--torq-disable-host
```

```text
diagnosis: no-npu-lowering
  fallback drop --torq-disable-css: passes: executor css
  fallback drop --torq-disable-host: passes: executor host
```

Remove the disable flag for the fallback you want to use. To allow CSS while keeping Host disabled:

```bash
torq-lab compile erf_f32.onnx --compiler-option=--torq-disable-host
```

You can also explicitly assign supported fallback operations to CSS:

```bash
torq-lab compile erf_f32.onnx \
    --compiler-option=--torq-disable-host \
    --compiler-option=--torq-execute-on-css=erf
```

The force-executor options accept `div`, `erf`, `reciprocal`, `rsqrt`, `softmax`, and `sqrt`. Use a comma-separated list to assign several operation types.

For assignments to individual operations, use an executor map from [torq-gen-config](torq-gen-config.md). Pass an absolute map path because relative compiler paths are resolved from the work directory:

```bash
torq-lab compile model.onnx \
    --compiler-option=--torq-executor-map="$PWD/model_compiler.json"
```

If the map assigns an operation to an executor you disabled, enable that executor or change the assignment. If the model must run entirely on the NPU, check the [supported operation constraints](ops.md) and change the dtype, shape, or operation accordingly.

### memory

The model exceeds LRAM or NSS memory limits. Memory recovery is always enabled. Try the NSS program space probe:

```bash
torq-lab debug model.mlir --probe
```

If the probe passes, increase NSS program space from 8 MB to 16 MB when compiling:

```bash
torq-lab compile model.mlir --compiler-option=--torq-max-nss-programs-size=16777216
```

If the probe fails, reduce or split the operation, use a smaller dtype, or choose a target with more memory. Memory failures skip the second compile used to collect IR, so a crash reproducer may be absent.

### dynamic-shape

The compiler requires static dimensions in the model's entry inputs and outputs. The report identifies the dynamic type, such as `!torch.vtensor<[1,?],f32>`.

Re-export ONNX models with fixed input and output shapes. For TFLite, convert to the model's default shapes and compile the result:

```bash
torq-lab convert_static tflite -i model.tflite -o model_static.tflite
torq-lab compile model_static.tflite
```

The converter requires the `torq-compiler[tf]` extra. This failure occurs before the compiler runs; share the model itself when asking for help.

### frontend-unsupported-op

The frontend cannot lower an ONNX operation. Read `source op` to find its ONNX name; `torch.operator` alone does not identify it.

Check the model's opset. Re-export using an equivalent supported operation, or request support for the reported operation. Use `--bisect` to identify affected layers in a larger model:

```bash
torq-lab debug model.onnx --bisect
```

### assert

The compiler aborted on a C++ assertion. Read the hint: some assertions describe a known unsupported configuration with a workaround; others need a compiler bug report.

For an `f32` convolution that requires `bf16` input and weights, convert the model and recompile:

```bash
torq-lab convert_dtype onnx -i model.onnx -o model_bf16.onnx -d bf16 --convert-io
torq-lab compile model_bf16.onnx
```

For an unfamiliar assertion, include the assertion's `file:line`, the model, and the reproduction bundle in an issue.

### invalid-model

ONNX validation rejected the model before compilation. The excerpt names the invalid node or input. Fix or re-export the model, then run `debug` again. Share the original model and `debug/raw-stderr.txt`; no compiler reproduction command is available for this failure.

### import-failed

The ONNX or TFLite importer failed. Read the excerpt or `debug/raw-stderr.txt` for the importer's error. Check the model's opset and install any missing dependency named in the message:

```bash
pip install "torq-compiler[onnx]"
# For TFLite models:
pip install "torq-compiler[tflite]"
```

## Share a failure report

Use a named work directory to make the files easy to find:

```bash
torq-lab debug model.onnx --work-dir model-debug
```

The report lists the files it created under `model-debug/debug/`:

| File | Use |
|------|-----|
| `diagnosis.json` | Structured diagnosis, probe results, tool version, and target information. |
| `raw-stderr.txt` | Full error output from the diagnostic compile. |
| `repro.sh` | Failing compiler command and a standalone crash-reproducer command when available. |
| `crash-repro.mlir` | Compiler-generated reproduction input, when available. |
| `failed-pass.mlir` | IR for an unsupported NPU operation, when available. |

Include the source model, the reported files, and any executor map referenced by the command. For ONNX and TFLite, include the imported `<work-dir>/<model-stem>.mlir` if `repro.sh` references it. The script uses paths from the original machine; update them before running it elsewhere.

To inspect the structured report or retry the compiler command locally:

```bash
python -m json.tool model-debug/debug/diagnosis.json
bash model-debug/debug/repro.sh
```

The script is expected to fail again. It can run both the original command and a standalone reproducer. Failures before compilation have no `repro.sh`. Check `metadata.crash_repro_parse` in `diagnosis.json` before relying on a crash reproducer.

For scripts, request JSON on stdout:

```bash
torq-lab debug model.onnx --json > diagnosis.json
```

Exit codes are `0` for success, `1` for a diagnosed failure, and `2` for invalid usage.

## Other diagnosis classes

### environment

A required tool is missing or the CSS configuration in `--torq-hw` is invalid. Install the named tool, check `TORQ_TOOL_PATH`, or correct the CSS configuration name.

### fatal-error

`LLVM ERROR:` or `UNREACHABLE executed` indicates a compiler failure. Report it with the stack dump and reproduction bundle.

### invalid-source

The input MLIR does not parse with this compiler build. Regenerate it from the model, especially if it was produced by an older compiler.

### kernel-unsupported

The selected TorqHW kernel cannot handle the operation's configuration. Check the [supported operation constraints](ops.md) and adjust the dtype, stride, or shape named in the error.

### serialization

The backend could not serialize the generated code. Inspect the earlier error lines in `raw-stderr.txt` for the underlying cause. Share the bundle if there is no actionable model constraint.

### tiling

The compiler could not tile, slice, or peel an operation. Try the slicing probe:

```bash
torq-lab debug model.mlir --probe
```

If it passes, compile with `--compiler-option=--torq-disable-slicing`.

### input-conversion

The input conversion cannot lower an operation or preserve its dtype. Adjust the I/O types or quantization named in the message.

### fallback-failed

CSS or Host is enabled but cannot build the operation. Read the following LLVM, linker, or allocation error in `raw-stderr.txt` to identify the fallback problem.

### legalization

A conversion pass could not lower the named operation. Inspect the failed-pass IR using the flags in [Debug tips and tricks](debug_tips.md), or share the bundle.

### non-convergence

Compiler rewrites did not converge. Report the model and bundle. Compiler developers can trace the rewrites with `--debug-only=greedy-rewriter`.

### timeout

The tool exceeded `--timeout`. The reported pass is the last one started before the timeout. Increase the timeout if the model normally takes longer; otherwise report the model, pass, and stderr. Further compiler probes are skipped after a timeout.

### silent-pass-failure

A pass failed without a useful diagnostic. Share the reported pass name and bundle. See [Debug tips and tricks](debug_tips.md) for flags that collect more IR.

### crash

The compiler exited on a signal without a recognized diagnostic. Report the model and bundle.

### unknown

No known failure signature matched. Start with the message and stderr, try `--probe`, and use `--bisect` for ONNX models. If the cause remains unclear, open an issue with the model and bundle.

## Further reading

### The manual ladder

Use these steps to investigate an unfamiliar failure or prepare a smaller reproduction for compiler developers. Stop once you have the cause and a useful reproduction input.

1. **Reproduce and capture.** Keep the exact command from `debug/repro.sh`, full stderr, and exit status. Re-run with `--mlir-disable-threading` if interleaved diagnostics obscure the failure. Keep the original result too: changing threading can change the failure.
2. **Identify the tool and stage.** Distinguish model validation, ONNX/TFLite import, the static-shape gate, frontend lowering, input conversion, Torq codegen, and serialization. The first useful error line and the [class index](#diagnosis-class-index) help locate the stage.
3. **Identify the failure type.** In a Python subprocess result, a negative return code indicates a signal: `-6` is an abort and `-11` is a segmentation fault. The lab's `ToolError` uses `None` for a timeout. Return code 1 with `error:` lines is a diagnostic failure; a translation wrapper alone may indicate a silent pass failure. Read the text as well, since crash-reproducer instrumentation can turn an abort into return code 1.
4. **Find the pass.** Start with `--mlir-print-ir-after-failure` and its `IR Dump After <Pass> Failed (<name>)` line. Use `--torq-show-pass-progress` to find the last started pass with no completion line. For nested codegen failures, use `--mlir-pass-pipeline-crash-reproducer=<file>` to capture the failing pipeline.
5. **Find the operation and source node.** Follow the diagnostic's `file:line:column` or named `loc(...)`. For ONNX imports, read that line in `<work-dir>/<model-stem>.mlir` for `torch.operator "onnx.<Op>"`. If the location is fused, inspect the failed-pass IR for operations that remain outside the target dialect.
6. **Minimize.** For ONNX, start with `--bisect`, then narrow the failure with subgraph extraction. If all checked layers pass but the full model fails, investigate fusion, tiling, or memory. For MLIR, capture phase dumps with `--dump-compilation-phases-to` and use `--compile-from=executable-sources` for a codegen reproduction.
7. **Change one option at a time.** A passing run narrows the path involved. Try `--torq-disable-slices` for the NSS path, remove `--torq-disable-css` or `--torq-disable-host` to allow fallback, use `--torq-disable-slicing` for slicing, `--torq-tile-and-fuse-producers-fuse-mode=only-patterns` for producer fusion, `--torq-max-nss-programs-size=16777216` for NSS program space, or `--torq-disable-dispatch-fusion` for dispatch formation. Verify any resulting model before adopting a workaround.
8. **Record the result.** Share the minimized input, failing command, relevant pass and source location, and any verified workaround. For a compiler bug, attach the reproduction bundle to the issue.

The compiler flags used here are documented in [Debug tips and tricks](debug_tips.md).

### Adding a class (for developers)

`RULES` in `torq.lab.debug` contains `Rule(class, stage, regex, hint)` entries; the class name is also the anchor of its heading in this guide. The first matching entry wins.

1. Capture a real failure's decisive stderr lines, return code, compiler options, and build information in `tests/_lab_debug_captures.py`. Keep diagnostic lines exact, mark omitted blocks, replace machine paths with `<<SRC>>` and `<<BUILD>>`, and label synthetic output.
2. Add a specific signature before broader rules. Timeout takes precedence over text, assertions precede pipeline errors, and crash requires a signal without recognized diagnostic text. Test overlapping signatures through their classification results.
3. Add the capture and expected classification to `CASES` in `tests/test_lab_debug.py`. Use return code 1 for a `LabError` raised before the compiler runs. Reuse existing scenarios for checks already covered by the suite.
4. Write an actionable hint and add a matching heading and index entry to this guide. Add an option to `PROBE_MAP` when it helps isolate the failure. Give common classes a worked example.
