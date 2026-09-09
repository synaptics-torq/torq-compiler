# torq-gen-config: a TORQ config generation tool

A guide for finding the best execution configuration (NSS/CSS/Host) for each operation in a model.

---

## Table of Contents

1. [Overview](#1-overview)
   - [Installation](#installation)
2. [Three-Step Workflow](#2-three-step-workflow)
3. [Viewing Results](#3-viewing-results)
   - [How Recommended Executor is Determined](#how-recommended-executor-is-determined)
   - [Changing the Recommended Executor](#changing-the-recommended-executor)
   - [Timing-Based Executor Recommendation](#timing-based-executor-recommendation)
4. [Handling Issues](#4-handling-issues)
5. [Auto-Converting FP32 Models to BF16](#5-auto-converting-fp32-models-to-bf16)
6. [ONNX to MLIR Mapping Mechanism](#6-onnx-to-mlir-mapping-mechanism)
7. [TFLite Model Support](#7-tflite-model-support)
8. [Command Reference](#8-command-reference)

---

## 1. Overview

### Why torq-gen-config?

TORQ has three executors, each with different strengths:

| Executor | Description | Priority |
|----------|-------------|----------|
| **NSS** | NPU Subsystem (hardware) | 1st |
| **CSS** | CPU Subsystem (hardware) | 2nd |
| **Host** | CPU fallback | 3rd |

Different operations work better on different executors. For example, convolution layers often run efficiently on NSS, while some complex tensor operations may only work correctly on Host. Additionally, the same operation might work on one executor but fail on another due to hardware limitations, unsupported data types, or numerical precision issues. torq-gen-config automatically tests each operation with all executors and records which ones work correctly, ensuring optimal performance and correctness when running the full model.

### How It Works

```
Model (ONNX) → Extract Layers → Test NSS/CSS/Host → Get Recommended Executor → Save JSON → Run Full Model
```

`torq-gen-config` accepts ONNX (`.onnx`) models and runs discovery fully
standalone: `discover` and `run` call `torq-compile` / `torq-run-module`
directly in-process, with no source checkout required. TFLite (`.tflite`)
discovery has been removed — the tool is ONNX-only; see
[TFLite Model Support](#7-tflite-model-support).

### Installation

Install the compiler and runtime wheels with `pip`; the `onnx` extra pulls in
the ONNX dependencies that discovery needs:

```bash
pip install "torq_compiler-<version>-<platform>.whl[onnx]"
pip install torq_runtime-<version>-<platform>.whl
```

This puts the `torq-gen-config` command on your `PATH`, ready to use — no
source checkout or environment setup is required. See
[Getting Started](getting_started.md) for all installation options.

### Example: SqueezeNet 1.0

This manual uses `squeezenet1.0-12.onnx` (66 operations: Conv, Relu, MaxPool, Concat, etc.) as a running example. It is available from the [ONNX model zoo](https://github.com/onnx/models/tree/main/validated/vision/classification/squeezenet).

### Tested Models

The following CNN models have been verified to work with torq-gen-config:

| Model | Type | Status |
|-------|------|--------|
| `alexnet_Opset16.onnx` | CNN | Working |
| `efficientnet_b1_Opset17_timm.onnx` | CNN | Working |
| `squeezenet1.0-12.onnx` | CNN | Working |
| `mobilenetv2-7.onnx` | CNN | Working |
| `resnet18_Opset18_timm.onnx` | CNN | Working |

### Output Files

Discovery produces two JSON files:

| File | Suffix | Purpose |
|------|--------|---------|
| **Report JSON** | `torq_gen_config_<model>.json` | Human-readable discovery results: statuses, timing, tolerances, per-layer details. Used by `view` and `edit`. |
| **Compiler JSON** | `torq_gen_config_<model>_compiler.json` | Minimal `op_assignments` consumed by the C++ `ExecutorAssignmentPass`. Auto-generated from the report JSON. |

Key rules:

1. Report JSON is the source of truth.
   If it exists, `run` always regenerates the compiler JSON from it.

2. `edit` only touches report JSON.
   It never reads or writes compiler JSON directly.

3. Compiler JSON is always a derived artifact.
   It's either regenerated from report JSON (on `run`) or hand-edited.

4. Compiler JSON alone is a valid input.
   If report JSON is absent, the compiler consumes the compiler JSON as-is.

5. Hand-edited compiler JSON is overwritten on the next `run` — unless
   the report JSON is missing, in which case your edits stick forever.

You normally only interact with the **report JSON**.

---

## 2. Three-Step Workflow

### Runtime Requirements

`torq-gen-config discover` / `run` run in-process and invoke the
`torq-compile` and `torq-run-module` binaries directly. When the compiler and
runtime wheels are installed, the binaries are found automatically inside the
installed packages — no further setup is needed. For non-standard
installations the binaries are located with the following precedence:

1. Environment variables pointing at the binaries: `TORQ_COMPILE`,
   `TORQ_RUN_MODULE` (plus `IREE_COMPILE` / `IREE_RUN_MODULE` for the
   llvm-cpu reference fallback).
2. `TORQ_TOOL_PATH`: extra directories to search, delimited like `PATH`.
3. Installed wheels: `torq-compile` ships in the compiler wheel
   (`torq/_compiler_libs/`), `torq-run-module` in the runtime wheel
   (`torq/_runtime_libs/`).
4. A development checkout: the sibling build tree (`$IREE_BUILD_DIR`, or
   `../iree-build` next to the repo).
5. The system `PATH`.

Intermediate artifacts (imported MLIR, compiled models, reference outputs)
live in a content-versioned cache:
`${XDG_CACHE_HOME:-~/.cache}/torq-gen-config/` by default, overridable with
the `TORQ_GEN_CONFIG_CACHE_DIR` environment variable. Cached artifacts are
reused across runs; `--recompute-cache` forces a rebuild, and deleting the
directory starts completely fresh.

The target hardware and runtime are selected with `--torq-hw TARGET`
(default: `default` = SL2610) and `--torq-hw-type TYPE` (default: `sim`,
forwarded to `torq-run-module --torq_hw_type`). Additional flags can be
passed through to the underlying tools with repeated `--compiler-option OPT`
/ `--runtime-option OPT`.

### Step 1: Discover

Test each operation to find the best executor:

```bash
torq-gen-config discover \
    --model squeezenet1.0-12.onnx \
    --output-dir ./results \
    --skip-mode
```

**What it does:**
- Extracts each layer from the model
- Tests NSS → CSS → Host
- Stops after first success (with `--skip-mode`)
- Creates `torq_gen_config_squeezenet1.0-12.json`

### Step 2: Review

View the JSON results with the built-in viewer:

```bash
# Using --model shortcut (auto-resolves JSON path from --output-dir)
torq-gen-config view \
    --model squeezenet1.0-12.onnx \
    --output-dir ./results

# Or specify the JSON path directly
torq-gen-config view ./results/torq_gen_config_squeezenet1.0-12.json
```

Output shows each layer with status for all executors:
```
Layer                                    NSS          CSS          HOST         Recommended
----------------------------------------------------------------------------------------------------
Conv_conv1_1                             success      difference   success      nss
Relu_conv1_2                             success      -            -            nss
MaxPool_pool1_1                          success      -            -            nss
...
```

Status types:
| Status | Meaning | Can Use? |
|--------|---------|----------|
| `success` | Works correctly, output matches reference | Yes |
| `difference` | Runs but output differs from reference (within tolerance) | Yes, with adjusted tolerance |
| `error` | Compilation or runtime failure | No |

**Understanding `difference` status:**

A `difference` status means the executor runs successfully but produces numerically different results. This often happens with BF16 models or hardware approximations. You can accept the difference by adjusting tolerance:

1. **View the current difference:**
   ```bash
   torq-gen-config view torq_gen_config_squeezenet1.0-12.json Conv_conv1_1
   ```

2. **Use the `edit` command** to increase tolerance:
   ```bash
   torq-gen-config edit \
       --model model.onnx \
       --layer Conv_conv1_1 \
       --tolerance-avg 0.1 \
       --tolerance-max 0.1
   ```

3. **Re-test with the new tolerance:** re-run discovery with
   `--recompute-cache`, which forces the layer to be recompiled and re-run:
   ```bash
   torq-gen-config discover \
       --model model.onnx \
       --output-dir ./results \
       --recompute-cache
   ```

If the layer passes with the new tolerance, the `recommended_executor` will be updated to prefer that executor.

### Step 3: Run Full Model

Compile and run the complete model with discovered assignments:

```bash
torq-gen-config run \
    --model squeezenet1.0-12.onnx \
    --output-dir ./results \
    --debug-ir=tmp
```

---

## 3. Viewing Results

### Using the Viewer Script

The viewer displays results layer by layer:

```bash
# Using --model shortcut (auto-resolves JSON path from --output-dir)
torq-gen-config view --model squeezenet1.0-12.onnx --output-dir results/

# Or specify the JSON path directly
torq-gen-config view results/torq_gen_config_squeezenet1.0-12.json

# View details for a specific layer
torq-gen-config view --model squeezenet1.0-12.onnx --output-dir results/ Conv_conv1_1
```

### Viewing Compiler JSON

The viewer also handles compiler-format JSON (`*_compiler.json`) gracefully:

```bash
# View compiler JSON — shows executor distribution and line:col assignments
torq-gen-config view --model squeezenet1.0-12.onnx --output-dir results/
# (auto-detects compiler format if report JSON is absent)

# Or point directly at the compiler JSON
torq-gen-config view torq_gen_config_squeezenet1.0-12_compiler.json
```

Output for a compiler JSON:
```
============================================================
MODEL: squeezenet1.0-12  (compiler format)
============================================================

Total assignments: 66

Executor distribution:
  NSS: 52
  CSS: 10
  HOST: 4

Assignments:
  42:12 → nss
  43:12 → nss
  ...
```

### Understanding the JSON

```json
{
  "ops": {
    "Conv_conv1_1": {
      "executors": {
        "nss": {"status": "success"},
        "css": {"status": "difference"},
        "host": {"status": "success"}
      },
      "recommended_executor": "nss",
      "_node_index": 0,
      "mlir_location": "271:12"
    }
  },
  "discovery_report": {
    "summary": {"total_layers": 66, "status_counts": {...}},
    "critical_failures": []
  }
}
```

### How Recommended Executor is Determined

The `recommended_executor` is selected automatically based on executor priority and test results:

**Priority Order:** `nss` → `css` → `host`

**Selection Logic:**
| Priority | Status | Example |
|----------|--------|---------|
| 1st | First executor with `success` | NSS success → recommend NSS |
| 2nd | First executor with `difference` | NSS error, CSS difference → recommend CSS |
| 3rd | Fallback to `host` | NSS/CSS error → recommend Host |

**Example scenarios:**
- NSS=`success`, CSS=`difference`, Host=`success` → recommends `nss` (highest priority success)
- NSS=`error`, CSS=`difference`, Host=`success` → recommends `css` (first working)
- NSS=`error`, CSS=`error`, Host=`success` → recommends `host` (only option)
- NSS=`error`, CSS=`error`, Host=`error` → **no recommendation (critical failure - full model cannot run!)**

### Changing the Recommended Executor

Use the `edit` command to safely override the recommendation. The command validates the JSON, updates the report, and regenerates the compiler config automatically.

**Edit a single layer:**
```bash
torq-gen-config edit \
    --model squeezenet1.0-12.onnx \
    --layer Conv_conv1_1 \
    --executor css
```

**Layer matching:** The `--layer` argument supports several strategies:

| Strategy | Example | Matches |
|----------|---------|---------|
| Exact (case-insensitive) | `--layer Conv_conv1_1` | `Conv_conv1_1` only |
| Substring | `--layer conv1` | All layers containing "conv1" |
| fnmatch wildcard | `--layer Conv_*` | All layers starting with "Conv_" |
| ALL | `--layer ALL` | Every layer |

**Batch edit multiple layers:**
```bash
# All Conv layers → NSS
torq-gen-config edit --model model.onnx --layer "Conv_*" --executor nss

# Every layer → CSS
torq-gen-config edit --model model.onnx --layer ALL --executor css
```

**Edit tolerance:**
```bash
torq-gen-config edit \
    --model model.onnx \
    --layer Conv_conv1_1 \
    --tolerance-avg 0.1 \
    --tolerance-max 0.5
```

**List available layers:**
```bash
torq-gen-config edit --model model.onnx --list
torq-gen-config edit --model model.onnx --list conv
```

**Notes:**
- The `edit` command only modifies the **report JSON** (`torq_gen_config_*.json`). It never reads or writes the compiler JSON directly.
- If you accidentally pass the compiler JSON (e.g., `torq_gen_config_model_compiler.json`), `edit` detects it and refuses, pointing you to the correct report JSON file.
- After editing, run `torq-gen-config run` — it will regenerate the compiler JSON from the updated report JSON before compiling.
- No need to re-run discovery. The compiler reads `recommended_executor` from the JSON at compile time.

### Timing-Based Executor Recommendation

By default, the recommended executor is determined by priority order (`nss` → `css` → `host`). However, you can use timing data to recommend the fastest executor instead.

**How it works:**
- When `--collect-timing` is enabled, runtime performance is measured for each executor
- With `--recommend-by-timing`, the fastest executor (lowest `runtime_ms`) with `success` status is recommended
- If no executor has `success` status, the fastest with `difference` status is recommended

**Usage:**

```bash
# Collect timing and recommend by performance
torq-gen-config discover \
    --model ./model.onnx \
    --collect-timing \
    --timing-runs=5 \
    --recommend-by-timing
```

**When to use:**
- When you want optimal performance rather than just functional correctness
- When multiple executors work but you want the fastest one
- For performance tuning and benchmarking

**Note:** `--recommend-by-timing` requires `--collect-timing` to be effective. If timing data is not available, falls back to priority-based selection.

---

## 4. Handling Issues

### Critical Failures (All Executors Failed)

A **critical failure** occurs when ALL three executors fail for an operation:

```
CRITICAL FAILURES (all executors error): 1
  - ProblematicOp_output (node: 42)
```

**Important:** If any operation has a critical failure, the **full model cannot run at all**. Every operation must have at least one working executor for the full model to compile and execute.

**Solutions (in order):**
1. **Debug specific layer** (see below) - understand why it's failing
2. **Use subgraph mode** - isolate the problematic operation and surrounding context
3. **Skip crashing executors** - if NSS/CSS hang, test with Host only
4. **Report issue** - if Host also fails, this may be an unsupported operation

### Debugging a Specific Layer

When a layer fails (e.g., CSS shows `error`), re-run discovery to see the
detailed per-executor error output. Pass `--recompute-cache` so cached
artifacts are rebuilt, and omit `--skip-mode` so every executor is exercised
even if one already succeeded:

```bash
torq-gen-config discover \
    --model squeezenet1.0-12.onnx \
    --output-dir ./results \
    --recompute-cache
```

To isolate a problematic region instead of re-running the whole model, use
subgraph mode (see [Subgraph Debugging](#subgraph-debugging)).

**Important: Discovery is for Executor Selection Only**

`torq-gen-config discover` tests which executor works for each operation but does NOT perform C++ executor assignment.

To see executor assignment in the IR dump, use **subgraph discovery** or a **full model run**:

```bash
# Subgraph discovery - shows executor assignment for that subgraph
torq-gen-config discover \
    --model model.onnx \
    --subgraph-from=Conv_0 \
    --subgraph-to=Conv_0

# Full model run - shows executor assignment for all operations
torq-gen-config run --model model.onnx --debug-ir=tmp
```

In the dumped IR:
```mlir
// With executor assignment
linalg.conv_2d_nchw_fchw {...} {torq-executor = "nss"}
```

**Complete Workflow:**

1. **First discovery** (get initial results):
   ```bash
   torq-gen-config discover --model model.onnx --skip-mode
   ```

2. **Re-test layers** (optional):
   ```bash
   # Re-run discovery; --recompute-cache forces layers to be rebuilt and
   # re-tested, and the JSON is updated with the new results
   torq-gen-config discover --model model.onnx --recompute-cache
   ```

3. **Verify executor assignment** (use subgraph discovery or a full model run — per-layer results won't show assignment):
   ```bash
   # Subgraph discovery shows executor assignment in IR
   torq-gen-config discover \
       --model model.onnx \
       --subgraph-from=Conv_0 \
       --subgraph-to=Conv_0
   
   # Or full model run
   torq-gen-config run --model model.onnx --debug-ir=tmp
   ```

**Example scenario - CSS error on Conv_0:**

1. View the error in JSON:
   ```bash
   torq-gen-config view \
       torq_gen_config_squeezenet1.0-12.json Conv_conv1_1
   ```

2. Re-run discovery to reproduce the failure with full output:
   ```bash
   torq-gen-config discover \
       --model squeezenet1.0-12.onnx \
       --output-dir ./results \
       --recompute-cache
   ```

3. If CSS consistently fails, use the `edit` command to switch to a different executor:
   ```bash
   torq-gen-config edit \
       --model model.onnx \
       --layer Conv_conv1_1 \
       --executor nss
   ```

4. To test without any executor assignment (debug mode), set executor to `null`:
   ```bash
   torq-gen-config edit \
       --model model.onnx \
       --layer Conv_conv1_1 \
       --executor null
   ```

### Subgraph Debugging

Subgraph mode tests a range of operations as a mini full-model. Use it to isolate problematic operations or debug specific model sections.

**Find operation names:**
```bash
python3 -c "
import onnx
model = onnx.load('squeezenet1.0-12.onnx')
for i, n in enumerate(model.graph.node):
    print(f'{i}: {n.op_type}_{n.output[0]}')
"
```

**Run subgraph discovery (nodes 10-16):**
```bash
torq-gen-config discover \
    --model squeezenet1.0-12.onnx \
    --subgraph-from=Conv_fire3/squeeze1x1_1 \
    --subgraph-to=Concat_fire3/concat_1 \
    --skip-mode
```

This creates `torq_gen_config_squeezenet1.0-12_subgraph_10_16.json` and runs layer discovery + full subgraph test.

**Subgraph Options:**
| Option | Description |
|--------|-------------|
| `--subgraph-from` | Start operation name |
| `--subgraph-to` | End operation name |

### Full Model Issues

If full model fails but individual layers pass:

1. Check debug IR: `ls tmp/`
2. Verify assignments in viewer: `torq-gen-config view torq_gen_config_*.json`
3. Re-run discovery for problematic layers

### Skipping Executors (Extra Debug Option)

If an executor crashes or hangs during discovery, skip it. Note that CSS runs
on the cycle simulator, so it is by far the slowest of the three — skipping it
is the easiest way to speed discovery up:

```bash
# Skip NSS only
torq-gen-config discover --model model.onnx --skip-executors=nss

# Skip CSS only (much faster discovery)
torq-gen-config discover --model model.onnx --skip-executors=css

# Skip both NSS and CSS (test only Host)
torq-gen-config discover --model model.onnx --skip-executors=nss,css
```

This helps identify if an operation works on at least one executor when others are unstable.

### Important: How `--skip-mode` and JSON Cache Work

**Understanding the interaction between `--skip-mode`, JSON file, and test execution:**

#### 1. `--skip-mode` Behavior

When `--skip-mode` is enabled:
- **First run**: Tests each executor (NSS → CSS → Host) until one succeeds, then saves `"status": "success"` to JSON
- **Subsequent runs**: Checks JSON file first - if a layer already has `"status": "success"`, the case is **skipped entirely**

This is designed for **speeding up incremental discovery**, not for re-testing.

#### 2. Discovery vs Full Model Run

**Discovery (`torq-gen-config discover`):**
- Purpose: **Discover** which executor works for each operation
- Results are compared against reference outputs and recorded in the JSON
- JSON is updated with the results
- Does NOT perform C++ executor assignment (per-layer MLIR has different line numbers)

**Full model run (`torq-gen-config run`):**
- Purpose: **Run** the complete model with discovered assignments
- If `recommended_executor` exists and is not null in JSON, the C++ ExecutorAssignmentPass **will** assign that executor
- The full model runs end-to-end

#### 3. Re-Running Layers After a Successful Discovery

**Problem:** You want to re-test a layer, but:
- The layer is skipped because the JSON already records a `success` for it
- No executor assignment happens in the dumped IR
- The run seems to use cached results

**Root Cause:** `--skip-mode` reads the JSON file and skips layers with a recorded `"status": "success"`. Without `--skip-mode`, every layer is re-run.

**Solutions:**

1. **Option A: Omit `--skip-mode`** (recommended for debugging)
   ```bash
   # Re-runs all layers regardless of JSON status
   torq-gen-config discover --model model.onnx
   ```
   Add `--recompute-cache` to also rebuild the cached compiled artifacts.

2. **Option B: Remove the layer's entry from the JSON**

   Delete the layer's block under `"ops"` in the report JSON, then re-run
   with `--skip-mode` — layers without a recorded `success` are tested again.

3. **Option C: Delete the JSON file**
   ```bash
   rm torq_gen_config_*.json
   torq-gen-config discover --model model.onnx --skip-mode --recompute-cache
   ```

#### 4. Verifying Executor Assignment

To verify executor assignment in the IR:

1. **Use subgraph discovery or a full model run** (per-layer results don't show assignment):
   ```bash
   # Subgraph discovery
   torq-gen-config discover --model model.onnx --subgraph-from=Conv_0 --subgraph-to=Conv_0
   
   # Or full model run
   torq-gen-config run --model model.onnx --debug-ir=tmp
   ```

2. **Check the dumped IR** in `tmp/` - look for `torq-executor` attributes:
   ```mlir
   // Example: operation assigned to NSS
   linalg.conv_2d_nchw_fchw {...} {torq-executor = "nss"}
   ```

#### 5. Summary Table: When to Use What

| Scenario | Skip Mode? | Recompute Cache? | Action on JSON |
|----------|------------|------------------|----------------|
| First discovery | Yes | Yes | None (will be created) |
| Add more test data | Yes | No | None (append mode) |
| Re-test layers | No | Optional | None, or delete the layer's entry |
| Force a specific executor | No | Optional | Set `recommended_executor` to desired executor |
| Full model with new assignments | N/A | No | Edit `recommended_executor` fields |

**Key Takeaway:** `--skip-mode` + existing JSON with `"status": "success"` = skipped layers. Omit skip mode or remove the JSON entries to actually re-run them.

---

## 5. Auto-Converting FP32 Models to BF16

TORQ NSS accelerator has limited FP32 support and requires BF16 (bfloat16) input for many operations. CSS and Host executors generally support FP32. The torq-gen-config framework provides automatic FP32 to BF16 conversion with accuracy validation.

### Why Convert to BF16?

| Scenario | Action Required |
|----------|-----------------|
| NSS executor fails with FP32 error | Convert to BF16 |
| Model weights are FP32 | May need conversion for NSS compatibility |
| Running on CSS/Host only | FP32 usually works |

**Note:** BF16 conversion may introduce minor numerical differences. Always validate accuracy after conversion.

### What is BF16?

BF16 is a 16-bit floating point format with:
- 1 sign bit
- 8 exponent bits (same as FP32)
- 7 mantissa bits (vs 23 for FP32)

**Key characteristics:**
- Same dynamic range as FP32 (no overflow issues)
- ~50% memory bandwidth reduction
- Truncation conversion is fast (just drop lower 16 bits)
- Required for NSS accelerator on TORQ hardware

### The Conversion Process

When `--auto-convert-bf16` is enabled:

1. **Weight Conversion**: All FP32 weights/biases are converted to BF16
2. **Type Annotation**: Input/output/intermediate tensors are marked as BF16
3. **Accuracy Validation**: Errors are computed for each tensor

### Accuracy Evaluation Method

The conversion accuracy is evaluated using **bit-truncation comparison** (see `scripts/convert_onnx_to_bf16.py` in the source repository):

```
FP32 (32 bits) → BF16 (16 bits) → FP32 (for comparison)
```

**Metrics computed per tensor:**
- `max_error`: Maximum absolute difference
- `mean_error`: Mean absolute difference  
- `rmse`: Root mean square error
- `max_rel_error`: Maximum relative error

**Interpretation guidelines:**

| Max Error | Quality | Usability |
|-----------|---------|-----------|
| < 0.01 | Excellent | Typical for BF16, safe for all use cases |
| < 0.1 | Good | Acceptable for most inference tasks |
| < 1.0 | Fair | May affect some sensitive layers |
| >= 1.0 | Poor | Significant accuracy loss, review needed |

### Inference-Level Accuracy Check (Optional)

Beyond weight-level checks, the conversion script (in the source repository) can compare end-to-end inference:

```bash
python scripts/convert_onnx_to_bf16.py model.onnx model_bf16.onnx --compare-inference --num-samples 10
```

This runs both models with random inputs and compares outputs:
- Runs `num_samples` (default 5) random input comparisons
- Uses ONNX Runtime for both FP32 and BF16 inference
- Reports per-sample and aggregate error statistics

### Using BF16 with torq-gen-config

**Basic usage:**
```bash
torq-gen-config discover --model model.onnx --auto-convert-bf16 --skip-mode
```

**Key points:**
- The conversion happens automatically before layer extraction
- Cache is invalidated when `--auto-convert-bf16` changes (the cache keys on input content and options)
- No manual pre-conversion needed - the framework handles everything

### Batch Dimension Handling

The conversion script automatically fixes dynamic batch dimensions:
- Converts symbolic dimensions (e.g., "batch", "N", "?") to fixed size 1
- Required for accurate inference comparison
- Warning is printed for each modified input

### Saving Converted Models

To save the BF16 model for external use:
```bash
torq-gen-config discover --model model.onnx --auto-convert-bf16 --save-bf16-model=/path/to/output.onnx
```

### When to Use BF16 Conversion

**Use BF16 when:**
- NSS executor reports FP32 is not supported
- Running layer discovery with NSS executor enabled
- Model has large weights (memory bandwidth constrained)

**Avoid BF16 when:**
- Running on CSS/Host only (FP32 usually works)
- Model has operations sensitive to numerical precision
- Accuracy requirements are strict (< 0.01% error tolerance)

**Note:** If NSS fails with "FP32 not supported" or "data type not supported" errors, use `--auto-convert-bf16`. CSS and Host executors typically handle FP32 without conversion.

---

## 6. ONNX to MLIR Mapping Mechanism

This section explains how torq-gen-config maps ONNX operations to their corresponding line numbers in the MLIR generated by torch-mlir. This mapping is essential for the C++ compiler to assign the correct executor to each operation.

### torch-mlir Import Guarantees

From the [torch-mlir architecture documentation](https://github.com/llvm/torch-mlir/blob/main/docs/architecture.md):

> "The torch dialect is almost entirely in **1:1 correspondence with the JIT IR** -- this allows the importer to be extremely small"

The ONNX to torch-mlir import process provides these key guarantees:

1. **Sequential Import**: torch-mlir's `onnx_importer.py` iterates through ONNX nodes sequentially:
   ```python
   def import_all(self, func=True):
       """Imports all nodes topologically."""
       for node in self._gi.graph_proto.node:  # Sequential iteration
           self.import_node(node)               # One ONNX node → one MLIR op
   ```

2. **No Fusion During Import**: Each ONNX node becomes exactly one `torch.operator` in MLIR (except for special handlers like Constant nodes)

3. **Topological Order**: Both ONNX and torch-mlir rely on topological ordering:
   > "ONNX requires that graphs be sorted topologically and free of cycles, so we don't take any special steps to order them for dominance."

### Position-Based Matching

The mapping uses **position (index)** as the matching key:

| Source | What We Track |
|--------|---------------|
| ONNX | `node_index` - position in `graph.node` (excluding Constants) |
| MLIR | Line number of each `torch.operator` in order of appearance |

**Why Position-Based is Reliable:**
- Deterministic: Both use topological ordering
- Verifiable: Can check that op types match at each position
- Simple: No complex heuristics or fuzzy matching

**Example Mapping:**
```
ONNX Node[0]: Conv → MLIR Line 42:12
ONNX Node[1]: Relu → MLIR Line 43:12
ONNX Node[2]: Conv → MLIR Line 44:12
```

The JSON stores this mapping:
```json
{
  "ops": {
    "Conv_output_0": {
      "recommended_executor": "nss",
      "_node_index": 0,       // Position in ONNX
      "mlir_location": "42:12"  // Line in MLIR
    }
  }
}
```

### Quantized Models

When `--quantize` is used, the final quantized MLIR contains extra `QuantizeLinear` and `DequantizeLinear` wrapper ops around the real compute ops. In addition, activations such as `Relu` may be fused into neighboring `Conv` ops and disappear as separate MLIR operations.

To handle this, the mapping is built from the **final quantized MLIR** rather than the original ONNX node order:

1. Import the quantized ONNX model into MLIR.
2. Strip `QuantizeLinear`/`DequantizeLinear` wrappers.
3. Build a list of quantized compute ops (`Conv`, `MaxPool`, `Add`, etc.) in MLIR order.
4. Align this list with the original unquantized layer op-types using **Longest Common Subsequence (LCS)**.
   - LCS preserves order and skips non-matching entries.
   - Fused activations are skipped automatically.
   - Each remaining compute op is mapped to its correct quantized MLIR line.

The resulting compiler JSON therefore assigns executors only to compute-op line numbers, never to Q/DQ wrapper lines.

**Fused activations in the report JSON:**  
A fused activation such as `Relu` still appears in the discovery report JSON with a `recommended_executor` from its per-layer test, because per-layer quantization keeps it as a standalone op (`Q → Relu → DQ`). However, it is intentionally omitted from the compiler JSON: in the full quantized model there is no separate `Relu` line to assign, so the fused `Conv+Relu` is represented entirely by the `Conv` assignment.

**Comparison with quantization:**  
When `--quantize` is active, `torq-gen-config` still compares results using the
same FP32 tolerances as non-quantized models. The quantized reference is
produced by ONNX Runtime, and the observed output is produced by the TORQ
compiler; the comparison reports the usual max absolute/relative difference and
number of differing elements.

### Verification

torq-gen-config automatically verifies the mapping during test generation:
- Count check: ONNX and MLIR have the same number of non-Constant ops
- Type check: Op types match at each position
- Warning output if verification fails

You can manually verify any model with a script in the source repository:

```bash
python scripts/verify_onnx_import_order.py --model-path=./model.onnx
```

If you see warnings like `COUNT MISMATCH` or `OP TYPE MISMATCHES` during discovery, the torch-mlir import behavior may have changed.

---

## 7. TFLite Model Support

TFLite discovery has been removed: `torq-gen-config` supports ONNX models
only, and passing a `.tflite` model to `discover` or `run` is rejected with
an error message.

`view` and `edit` still work on previously generated TFLite report JSONs
exactly as they do for ONNX. TFLite layers are keyed by operator name and
index (for example `CONV_2D_0`, `DEQUANTIZE_1`), and those IDs are used
wherever a layer ID is expected.

---

## 8. Command Reference

### torq-gen-config CLI

The command-line interface to the discovery system, installed by the compiler
wheel. TFLite models are rejected by `discover` and `run` (TFLite discovery
has been removed); see
[TFLite Model Support](#7-tflite-model-support).

#### `discover` — Run executor discovery

| Option | Description |
|--------|-------------|
| `--model` | Path to ONNX (`.onnx`) model (**required**) |
| `--output-dir` | Directory for generated JSON (default: current directory) |
| `--torq-hw` | Target hardware passed to `torq-compile` (default: `default` = SL2610) |
| `--torq-hw-type` | Runtime hardware type passed to `torq-run-module --torq_hw_type` (default: `sim`) |
| `--compiler-option` | Extra option passed through to `torq-compile` (repeatable) |
| `--runtime-option` | Extra option passed through to `torq-run-module` (repeatable) |
| `--skip-mode` | Stop after first success per layer |
| `--skip-executors` | Comma-separated list to skip (e.g., `nss,css`) |
| `--auto-convert-bf16` | Convert FP32 model to BF16 |
| `--save-bf16-model` | Save converted BF16 model to path |
| `--subgraph-from` | Start op name for subgraph |
| `--subgraph-to` | End op name for subgraph |
| `--collect-timing` | Collect runtime timing data |
| `--timing-runs` | Number of runtime runs for timing average |
| `--recommend-by-timing` | Recommend fastest executor based on timing |
| `--dedup-layers` | Detect duplicate layers and copy results |
| `--quantize` | Quantize each layer to int8 before testing |
| `--per-channel` | Use per-channel weight quantization with `--quantize` |
| `--full-integer` | Rewrite quantized I/O to int8 (remove input Q/output DQ) |
| `--quant-format` | ONNX quantization format: `qdq` (default) or `qoperator` |
| `--recompute-cache` | Force recompute cached fixtures during discovery |
| `--log-file` | Tee discovery output to this log file; full per-layer error diagnostics (e.g. compiler stderr) are written here, keeping the console to one-line summaries |
| `-v`, `--verbose` | Show detailed logs (JSON cache activity, MLIR conversion, comparison metrics, skip reasons) |

```bash
# Basic discovery
torq-gen-config discover --model model.onnx

# With skip mode and BF16 conversion
torq-gen-config discover --model model.onnx --skip-mode --auto-convert-bf16

# Quantized discovery (QDQ)
torq-gen-config discover --model model.onnx --skip-mode --quantize

# Quantized discovery (full-integer int8 I/O)
torq-gen-config discover --model model.onnx --skip-mode --quantize --full-integer

# Quantized discovery using QOperator (native int8 ops such as QLinearConv).
# For Host/NSS it is recommended to also pass --full-integer so graph I/O are int8;
# QOperator without --full-integer may fail to legalize ops such as QLinearAdd.
torq-gen-config discover --model model.onnx --skip-mode --quantize --quant-format=qoperator --full-integer

# Timing-based recommendation
torq-gen-config discover --model model.onnx --collect-timing --timing-runs=5 --recommend-by-timing
```

#### `run` — Run full model test

| Option | Description |
|--------|-------------|
| `--model` | Path to ONNX (`.onnx`) model (**required**) |
| `--output-dir` | Directory where config JSON is located |
| `--torq-hw` | Target hardware passed to `torq-compile` (default: `default` = SL2610) |
| `--torq-hw-type` | Runtime hardware type passed to `torq-run-module --torq_hw_type` (default: `sim`) |
| `--compiler-option` | Extra option passed through to `torq-compile` (repeatable) |
| `--runtime-option` | Extra option passed through to `torq-run-module` (repeatable) |
| `--auto-convert-bf16` | Convert FP32 model to BF16 |
| `--quantize` | Quantize the full model to int8 before compiling |
| `--per-channel` | Use per-channel weight quantization with `--quantize` |
| `--full-integer` | Rewrite quantized I/O to int8 (remove input Q/output DQ) |
| `--quant-format` | ONNX quantization format: `qdq` (default) or `qoperator` |
| `--debug-ir` | Dump IR directory for debugging (default: `tmp`) |
| `--recompute-cache` | Force recompute cached fixtures |
| `--log-file` | Tee output to this log file; full error diagnostics (e.g. compiler stderr) are written here, keeping the console to one-line summaries |
| `-v`, `--verbose` | Show detailed logs (JSON cache activity, comparison metrics) |

```bash
# Run full model with discovered assignments
torq-gen-config run --model model.onnx

# With debug IR dump
torq-gen-config run --model model.onnx --debug-ir=tmp

# Run full model with the same quantization settings used during discovery
torq-gen-config run --model model.onnx --output-dir results/ --quantize
```

**Note:** `run` accepts either the report JSON or the compiler JSON. If the report JSON exists, `run` regenerates the compiler JSON from it before compiling. If only the compiler JSON exists, the full model test uses it directly. When quantization was used during discovery, pass the same `--quantize`/`--per-channel`/`--full-integer`/`--quant-format` flags to `run` so the full model is compiled with the same settings.

#### `view` — View executor config

| Argument | Description |
|----------|-------------|
| `config` | Path to report or compiler JSON (optional; `--model` auto-resolves) |
| `--model` | Path to ONNX model (auto-resolves JSON from model name) |
| `--output-dir` | Directory where config JSON is located (default: current directory) |
| `layer` | Optional layer ID for detailed view |

```bash
# Using --model shortcut
torq-gen-config view --model model.onnx --output-dir results/

# View summary from report JSON path
torq-gen-config view torq_gen_config_model.json

# View details for one layer
torq-gen-config view torq_gen_config_model.json Conv_conv1_1

# View compiler JSON (auto-detected)
torq-gen-config view torq_gen_config_model_compiler.json

# View layer details with --model
torq-gen-config view --model model.onnx --output-dir results/ Conv_conv1_1
```

#### `edit` — Edit executor assignments

| Option | Description |
|--------|-------------|
| `config` | Path to report JSON (optional; `--model` auto-resolves) |
| `--model` | Path to ONNX model (auto-resolves JSON from model name) |
| `--output-dir` | Directory where config JSON is located |
| `--layer` | Layer ID to edit. Supports exact, substring, fnmatch (`*`, `?`), or `ALL` |
| `--executor` | Set recommended executor (`nss`/`css`/`host`/`null`) |
| `--tolerance-avg` | Set `fp_avg_tol` for this layer |
| `--tolerance-max` | Set `fp_max_tol` for this layer |
| `--list [FILTER]` | List available layers and exit |

```bash
# Edit single layer
torq-gen-config edit --model model.onnx --layer Conv_0 --executor nss

# Batch edit all Conv layers
torq-gen-config edit --model model.onnx --layer "Conv_*" --executor nss

# Edit every layer
torq-gen-config edit --model model.onnx --layer ALL --executor css

# Update tolerance
torq-gen-config edit --model model.onnx --layer Conv_0 --tolerance-avg 0.1

# List layers
torq-gen-config edit --model model.onnx --list
torq-gen-config edit --model model.onnx --list conv
```
