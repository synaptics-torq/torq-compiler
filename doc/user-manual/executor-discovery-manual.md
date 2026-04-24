# TORQ Executor Discovery Manual

A guide for finding the best executor (NSS/CSS/Host) for each operation in a model.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Three-Step Workflow](#2-three-step-workflow)
3. [Viewing Results](#3-viewing-results)
   - [How Recommended Executor is Determined](#how-recommended-executor-is-determined)
   - [Changing the Recommended Executor](#changing-the-recommended-executor)
   - [Timing-Based Executor Recommendation](#timing-based-executor-recommendation)
4. [Handling Issues](#4-handling-issues)
5. [Auto-Converting FP32 Models to BF16](#5-auto-converting-fp32-models-to-bf16)
6. [ONNX to MLIR Mapping Mechanism](#6-onnx-to-mlir-mapping-mechanism)
7. [Command Options](#7-command-options)

---

## 1. Overview

### Why Executor Discovery?

TORQ has three executors, each with different strengths:

| Executor | Description | Priority |
|----------|-------------|----------|
| **NSS** | NPU Subsystem (hardware) | 1st |
| **CSS** | CPU Subsystem (hardware) | 2nd |
| **Host** | CPU fallback | 3rd |

Different operations work better on different executors. For example, convolution layers often run efficiently on NSS, while some complex tensor operations may only work correctly on Host. Additionally, the same operation might work on one executor but fail on another due to hardware limitations, unsupported data types, or numerical precision issues. Executor Discovery automatically tests each operation with all executors and records which ones work correctly, ensuring optimal performance and correctness when running the full model.

### How It Works

```
ONNX Model → Extract Layers → Test NSS/CSS/Host → Get Recommended Executor → Save JSON → Run Full Model
```

### Example: SqueezeNet 1.0

This manual uses `squeezenet1.0-12.onnx` (66 operations: Conv, Relu, MaxPool, Concat, etc.) as a running example.

### Tested Models

The following CNN models have been verified to work with executor discovery:

| Model | Type | Status |
|-------|------|--------|
| `alexnet_Opset16.onnx` | CNN | Working |
| `efficientnet_b1_Opset17_timm.onnx` | CNN | Working |
| `squeezenet1.0-12.onnx` | CNN | Working |
| `mobilenetv2-7.onnx` | CNN | Working |
| `resnet18_Opset18_timm.onnx` | CNN | Working |

---

## 2. Three-Step Workflow

### Step 1: Discover

Test each operation to find the best executor:

```bash
pytest tests/test_onnx_executor_discovery.py \
    -v -k "_layer_" \
    --model-path=./tests/testdata/onnx_models/squeezenet1.0-12.onnx \
    --executor-skip-mode --recompute-cache
```

**What it does:**
- Extracts each layer from the model
- Tests NSS → CSS → Host
- Stops after first success (with `--executor-skip-mode`)
- Creates `executor_assignments_squeezenet1.0-12.json`

### Step 2: Review

View the JSON results with the built-in viewer:

```bash
python3 python/torq/executor_discovery/view_discovery_json.py executor_assignments_squeezenet1.0-12.json
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
   python3 python/torq/executor_discovery/view_discovery_json.py executor_assignments_squeezenet1.0-12.json Conv_conv1_1
   ```

2. **Edit the JSON** to increase tolerance for that executor:
   ```json
   {
     "ops": {
       "Conv_conv1_1": {
         "executors": {
           "css": {
             "status": "difference",
             "tolerance_used": {"fp_avg_tol": 0.1, "fp_max_tol": 0.1}
           }
         }
       }
     }
   }
   ```

3. **Re-test just that layer:**
   ```bash
   pytest ... -k "squeezenet1.0-12_layer_Conv_0_css" --recompute-cache
   ```

If the test passes with new tolerance, the `recommended_executor` will be updated to prefer that executor.

### Step 3: Run Full Model

Compile and run the complete model with discovered assignments:

```bash
pytest tests/test_onnx_executor_discovery.py \
    -v -k "_full_model" \
    --model-path=./tests/testdata/onnx_models/squeezenet1.0-12.onnx \
    --debug-ir=tmp --recompute-cache
```

---

## 3. Viewing Results

### Using the Viewer Script

The viewer displays results layer by layer:

```bash
# View summary of all layers
python3 python/torq/executor_discovery/view_discovery_json.py executor_assignments_squeezenet1.0-12.json

# View details for specific layer
python3 python/torq/executor_discovery/view_discovery_json.py executor_assignments_squeezenet1.0-12.json Conv_conv1_1
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

You can manually override the recommendation by editing the JSON file:

1. **Open the JSON file:**
   ```bash
   vim executor_assignments_squeezenet1.0-12.json
   ```

2. **Change the `recommended_executor` field:**
   ```json
   {
     "ops": {
       "Conv_conv1_1": {
         "executors": {...},
         "recommended_executor": "css",
         "_node_index": 0,
         "mlir_location": "271:12"
       }
     }
   }
   ```

3. **Run full model test** - the compiler will use your updated assignment:
   ```bash
   pytest ... -k "_full_model" --model-path=./tests/testdata/onnx_models/squeezenet1.0-12.onnx
   ```

**Note:** The compiler reads `recommended_executor` from the JSON at compile time. No need to re-run discovery - just edit and recompile.

### Timing-Based Executor Recommendation

By default, the recommended executor is determined by priority order (`nss` → `css` → `host`). However, you can use timing data to recommend the fastest executor instead.

**How it works:**
- When `--collect-timing` is enabled, runtime performance is measured for each executor
- With `--recommend-by-timing`, the fastest executor (lowest `runtime_ms`) with `success` status is recommended
- If no executor has `success` status, the fastest with `difference` status is recommended

**Usage:**

```bash
# Collect timing and recommend by performance
pytest tests/test_onnx_executor_discovery.py \
    -v -k "_layer_" \
    --model-path=./model.onnx \
    --collect-timing \
    --timing-runs=5 \
    --recommend-by-timing \
    --recompute-cache
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

When a layer fails (e.g., CSS shows `error`), debug it individually:

```bash
# Re-test a specific layer with all executors
pytest tests/test_onnx_executor_discovery.py \
    -v -k "squeezenet1.0-12_layer_Conv_0" \
    --model-path=./tests/testdata/onnx_models/squeezenet1.0-12.onnx \
    --recompute-cache
```

This runs the layer (`Conv_0`) with NSS, CSS, and Host separately to see detailed error output.

**Important: Layer Tests are for Discovery Only**

Layer tests (`-k "_layer_"`) are designed for **executor discovery** only. They test which executor works for each operation but do NOT perform C++ executor assignment.

To see executor assignment in the IR dump, use **subgraph test** or **full model test**:

```bash
# Subgraph test - shows executor assignment for that subgraph
pytest ... --subgraph-from=Conv_0 --subgraph-to=Conv_0 -k "_full" --debug-ir=tmp

# Full model test - shows executor assignment for all operations
pytest ... -k "_full_model" --debug-ir=tmp
```

In the dumped IR:
```mlir
// With executor assignment
linalg.conv_2d_nchw_fchw {...} {torq-executor = "nss"}
```

**Complete Workflow:**

1. **First discovery** (get initial results):
   ```bash
   pytest ... -k "_layer_" --executor-skip-mode --recompute-cache
   ```

2. **Debug specific layers** (optional - set `recommended_executor: null` to re-test):
   ```bash
   # Edit JSON: set recommended_executor to null for the layer
   pytest ... -k "model_layer_Conv_0_nss" --recompute-cache
   # JSON automatically updated with new results
   ```

3. **Verify executor assignment** (use subgraph or full model - layer tests won't show assignment):
   ```bash
   # Subgraph test shows executor assignment in IR
   pytest ... --subgraph-from=Conv_0 --subgraph-to=Conv_0 -k "_full" --debug-ir=tmp
   
   # Or full model test
   pytest ... -k "_full_model" --debug-ir=tmp
   ```

**Example scenario - CSS error on Conv_0:**

1. View the error in JSON:
   ```bash
   python3 python/torq/executor_discovery/view_discovery_json.py \
       executor_assignments_squeezenet1.0-12.json Conv_conv1_1
   ```

2. Re-test that specific layer:
   ```bash
   pytest ... -k "squeezenet1.0-12_layer_Conv_0_css" --recompute-cache
   ```

3. If CSS consistently fails, edit the JSON to use a different executor:
   ```json
   {
     "ops": {
       "Conv_conv1_1": {
         "recommended_executor": "nss"
       }
     }
   }
   ```

4. To test without any executor assignment (debug mode):
   ```json
   {
     "ops": {
       "Conv_conv1_1": {
         "recommended_executor": null
       }
     }
   }
   ```

### Subgraph Debugging

Subgraph mode tests a range of operations as a mini full-model. Use it to isolate problematic operations or debug specific model sections.

**Find operation names:**
```bash
python3 -c "
import onnx
model = onnx.load('./tests/testdata/onnx_models/squeezenet1.0-12.onnx')
for i, n in enumerate(model.graph.node):
    print(f'{i}: {n.op_type}_{n.output[0]}')
"
```

**Run subgraph discovery (nodes 10-16):**
```bash
pytest tests/test_onnx_executor_discovery.py \
    -v \
    --model-path=./tests/testdata/onnx_models/squeezenet1.0-12.onnx \
    --subgraph-from=Conv_fire3/squeeze1x1_1 \
    --subgraph-to=Concat_fire3/concat_1 \
    --executor-skip-mode --recompute-cache
```

This creates `executor_assignments_squeezenet1.0-12_subgraph_10_16.json` and runs layer discovery + full subgraph test.

**Subgraph Options:**
| Option | Description |
|--------|-------------|
| `--subgraph-from` | Start operation name |
| `--subgraph-to` | End operation name |
| `-k "_layer_"` | Layer discovery only |
| `-k "_full"` | Full subgraph test only |

### Full Model Issues

If full model fails but individual layers pass:

1. Check debug IR: `ls tmp/`
2. Verify assignments in viewer: `python3 python/torq/executor_discovery/view_discovery_json.py executor_assignments_*.json`
3. Re-run discovery for problematic layers

### Skipping Executors (Extra Debug Option)

If NSS or CSS crashes/hangs during discovery, skip them:

```bash
# Skip NSS only
pytest ... --skip-executors=nss

# Skip both NSS and CSS (test only Host)
pytest ... --skip-executors=nss,css
```

This helps identify if an operation works on at least one executor when others are unstable.

### Important: How `--executor-skip-mode` and JSON Cache Work

**Understanding the interaction between `--executor-skip-mode`, JSON file, and test execution:**

#### 1. `--executor-skip-mode` Behavior

When `--executor-skip-mode` is enabled:
- **First run**: Tests each executor (NSS → CSS → Host) until one succeeds, then saves `"status": "success"` to JSON
- **Subsequent runs**: Checks JSON file first - if a layer already has `"status": "success"`, the test is **skipped entirely** (pytest.skip)

This is designed for **speeding up incremental discovery**, not for re-testing.

#### 2. Layer Test vs Full Model Test

**Layer Test (`-k "_layer_"`):**
- Purpose: **Discover** which executor works for each operation
- Test passes/fails based on comparison with reference results
- JSON is updated with test results
- Does NOT perform C++ executor assignment (layer MLIR has different line numbers)

**Full Model Test (`-k "_full_model"`):**
- Purpose: **Run** the complete model with discovered assignments
- If `recommended_executor` exists and is not null in JSON, the C++ ExecutorAssignmentPass **will** assign that executor
- The full model runs end-to-end

#### 3. Debugging Specific Layers - Common Pitfall

**Problem:** You want to debug a layer and check its executor assignment, but:
- Layer test shows "SKIPPED" even with `--recompute-cache`
- No executor assignment happens in the dumped IR
- The test seems to use cached results

**Root Cause:** `--executor-skip-mode` reads the JSON file and skips tests for layers with `"status": "success"`. The `--recompute-cache` only invalidates the ONNX/MLIR file cache, not the JSON test results.

**Solution - To actually re-run and check executor assignment:**

1. **Option A: Remove `--executor-skip-mode`** (recommended for debugging)
   ```bash
   # This will re-run all tests regardless of JSON status
   pytest ... -k "squeezenet1.0-12_layer_Conv_0" --recompute-cache
   # Note: WITHOUT --executor-skip-mode
   ```

2. **Option B: Set `recommended_executor` to `null`**
   ```json
   {
     "ops": {
       "Conv_conv1_1": {
         "recommended_executor": null,
         "executors": {
           "nss": {"status": "success"},
           "css": {"status": "success"},
           "host": {"status": "success"}
         }
       }
     }
   }
   ```
   Then run with `--executor-skip-mode` - it will test all executors again.

3. **Option C: Delete the JSON file**
   ```bash
   rm executor_assignments_*.json
   pytest ... -k "_layer_" --executor-skip-mode --recompute-cache
   ```

#### 4. Verifying Executor Assignment

To verify executor assignment in the IR:

1. **Use subgraph or full model test** (layer tests don't show assignment):
   ```bash
   # Subgraph test
   pytest ... --subgraph-from=Conv_0 --subgraph-to=Conv_0 -k "_full" --debug-ir=tmp
   
   # Or full model test
   pytest ... -k "_full_model" --debug-ir=tmp
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
| Re-test layer | No | Yes | Set `recommended_executor` to `null` or delete entry |
| Force test specific executor | No | Yes | Set `recommended_executor` to desired executor |
| Full model with new assignments | N/A | No | Edit `recommended_executor` fields |

**Key Takeaway:** `--executor-skip-mode` + existing JSON with `"status": "success"` = skipped tests. Remove skip mode or modify JSON to actually re-run tests.

---

## 5. Auto-Converting FP32 Models to BF16

TORQ NSS accelerator has limited FP32 support and requires BF16 (bfloat16) input for many operations. CSS and Host executors generally support FP32. The executor discovery framework provides automatic FP32 to BF16 conversion with accuracy validation.

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

The conversion accuracy is evaluated using **bit-truncation comparison** (see `scripts/convert_onnx_to_bf16.py`):

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

Beyond weight-level checks, the conversion script can compare end-to-end inference:

```bash
python scripts/convert_onnx_to_bf16.py model.onnx model_bf16.onnx --compare-inference --num-samples 10
```

This runs both models with random inputs and compares outputs:
- Runs `num_samples` (default 5) random input comparisons
- Uses ONNX Runtime for both FP32 and BF16 inference
- Reports per-sample and aggregate error statistics

### Using BF16 with Executor Discovery

**Basic usage:**
```bash
pytest tests/test_onnx_executor_discovery.py \
    -v -k "_layer_" \
    --model-path=./model.onnx \
    --auto-convert-bf16 \
    --executor-skip-mode --recompute-cache
```

**Key points:**
- The conversion happens automatically before layer extraction
- Cache is invalidated when `--auto-convert-bf16` changes (via versioned fixtures)
- No manual pre-conversion needed - the framework handles everything

### Batch Dimension Handling

The conversion script automatically fixes dynamic batch dimensions:
- Converts symbolic dimensions (e.g., "batch", "N", "?") to fixed size 1
- Required for accurate inference comparison
- Warning is printed for each modified input

### Saving Converted Models

To save the BF16 model for external use:
```bash
pytest ... --auto-convert-bf16 --save-bf16-model=/path/to/output.onnx
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

This section explains how executor discovery maps ONNX operations to their corresponding line numbers in the MLIR generated by torch-mlir. This mapping is essential for the C++ compiler to assign the correct executor to each operation.

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

### Verification

Executor discovery automatically verifies the mapping during test generation:
- Count check: ONNX and MLIR have the same number of non-Constant ops
- Type check: Op types match at each position
- Warning output if verification fails

You can manually verify any model:

```bash
python scripts/verify_onnx_import_order.py --model-path=./model.onnx
```

If you see warnings like `COUNT MISMATCH` or `OP TYPE MISMATCHES` during discovery, the torch-mlir import behavior may have changed.

---

## 7. Command Options

### Essential Options

| Option | Description |
|--------|-------------|
| `--model-path` | Path to ONNX model |
| `-k "_layer_"` | Run layer discovery |
| `-k "_full_model"` | Run full model test |
| `--executor-skip-mode` | Stop after first success per layer |
| `--recompute-cache` | Force recompute (ignore cache) |
| `--debug-ir=DIR` | Dump IR for debugging |

### Additional Options

| Option | Description |
|--------|-------------|
| `--skip-executors=nss,css` | Skip specific executors |
| `--auto-convert-bf16` | Convert FP32 to BF16 |
| `--subgraph-from=OP` | Subgraph start |
| `--subgraph-to=OP` | Subgraph end |
| `--collect-timing` | Collect compile and runtime timing data |
| `--timing-runs=N` | Number of runtime runs for timing average (default: 1) |
| `--recommend-by-timing` | Recommend fastest executor based on timing data |
| `--executor-discovery-log-file=PATH` | Redirect all output to log file; terminal shows only test progress and final report (use with `-s`) |
| `--dedup-layers` | Detect duplicate layers by ONNX signature and copy results instead of re-testing |

### Common Patterns

```bash
# Layer discovery with skip mode
pytest ... --model-path=model.onnx -k "_layer_" --executor-skip-mode

# Full model with debug output
pytest ... --model-path=model.onnx -k "_full_model" --debug-ir=tmp

# Subgraph debugging
pytest ... --model-path=model.onnx --subgraph-from=StartOp --subgraph-to=EndOp

# Skip crashing executors
pytest ... --model-path=model.onnx --skip-executors=nss -k "_layer_"

# Timing-based executor recommendation (performance optimization)
pytest ... --model-path=model.onnx -k "_layer_" --collect-timing --timing-runs=5 --recommend-by-timing

# Redirect output to log file (terminal shows only progress + final report)
pytest ... --model-path=model.onnx -k "_layer_" -v -s \
    --executor-discovery-log-file=discovery.log

# Skip duplicate layers by ONNX signature (speeds up models with repeated blocks)
pytest ... --model-path=model.onnx -k "_layer_" --dedup-layers --executor-skip-mode
```
