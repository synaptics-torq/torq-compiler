# TORQ ONNX Quantization

This document describes TORQ's ONNX integer quantization support.

## Table of Contents

1. [Overview](#1-overview)
   - [High-level lowering flow](#high-level-lowering-flow)
2. [ONNX Quantization Formats](#2-onnx-quantization-formats)
   - [QDQ](#qdq)
   - [QOperator](#qoperator)
   - [Hybrid](#hybrid)
3. [torq-gen-config Quantization Workflow](#3-torq-gen-config-quantization-workflow)
   - [Supported Quantization Options](#supported-quantization-options)
   - [Per-Layer vs. Full-Model Quantization](#per-layer-vs-full-model-quantization)
   - [CLI Usage](#cli-usage)
4. [Quantized Convolution Zero-Point Correction](#4-quantized-convolution-zero-point-correction)
5. [References](#5-references)

---

## 1. Overview

TORQ's ONNX quantization pipeline is built around
[ONNX Runtime (ORT) static quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html).
ORT takes a floating-point ONNX model, runs it over a representative dataset,
and records the numeric range of each activation. Those ranges become fixed
`scale` and `zero_point` values that are baked into the quantized graph.

TORQ wraps ORT quantization and integrates it into the `torq-gen-config`
discovery and run workflows. The resulting quantized ONNX model is imported to
MLIR, compiled with the TORQ compiler, and compared against an ONNX Runtime
reference running the same quantized model.

### High-level lowering flow

A quantized Conv2D can arrive from ONNX in either QDQ or QOperator form. The two
representations look different at the ONNX and Torch-MLIR levels, but both end
up at the same `linalg.conv_2d_nchw_fchw_q` and finally `torq_hl.conv2d`.

```
┌─────────────────────────┐     ┌─────────────────────────┐
│ ONNX QDQ graph          │     │ ONNX QOperator graph    │
│ (full-integer I/O)      │     │                         │
│                         │     │                         │
│ input_i8 → DQ → f32     │     │ input_i8 ──────────┐    │
│ weight_i8 → DQ → f32    │     │ weight_i8          │    │
│ Conv(f32) → f32         │     │ scale/zp inputs    │    │
│ f32 → Q → output_i8     │     │ QLinearConv → i8   │    │
└───────────┬─────────────┘     └───────────┬─────────────┘
            │ import_onnx                   │ import_onnx
            ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│ Torch-MLIR              │     │ Torch-MLIR              │
│ aten.dequantize         │     │ onnx.QLinearConv        │
│ aten.convolution (f32)  │     │                         │
│ aten.quantize_per_tensor│     │                         │
└───────────┬─────────────┘     └───────────┬─────────────┘
            │ torch-fuse-quantized-ops      │ (importer
            │ (fuses DQ/Conv/Q into         │  lowers directly)
            │  quantized aten.convolution)  │
            ▼                               ▼
┌─────────────────────────┐     ┌─────────────────────────┐
│ Torch-MLIR quantized    │     │ Torch-MLIR quantized    │
│                         │     │                         │
│ aten.convolution        │     │ aten.convolution        │
│ (qint8 input/weight,    │     │ (qint8 input/weight,    │
│  si32 output)           │     │  si32 output)           │
│ dequantize → quant      │     │ dequantize → quant      │
│ (output rescale)        │     │ (output rescale)        │
└───────────┬─────────────┘     └───────────┬─────────────┘
            │ convert-torch-to-linalg       │ convert-torch-to-linalg
            ▼                               ▼
┌─────────────────────────────────────────────────────────┐
│ Linalg quantized conv                                   │
│                                                         │
│ linalg.conv_2d_nchw_fchw_q(ins: i8, i8, i32 zp, i32 zp) │
│   → i32                                                 │
│ linalg.generic (re-quant i32 → i8)                      │
└───────────────────────────┬─────────────────────────────┘
                            │
                            │ LinalgToTorqHL (TORQ compiler)
                            ▼
            ┌───────────────────────────────┐
            │ torq_hl.conv2d                │
            │ (i8 input, i16 weight,        │
            │  i32 scale_bias) → i8         │
            └───────────────────────────────┘
```

At the ONNX level, QDQ keeps the original FP32 `Conv` and wraps it with
`QuantizeLinear`/`DequantizeLinear` nodes. QOperator replaces the op with a
native int8 op such as `QLinearConv`. After import:

- **QDQ** is imported as `aten.dequantize → aten.convolution(f32) →
  aten.quantize_per_tensor`. The `torch-fuse-quantized-ops` pass folds that
  sequence into a quantized `aten.convolution` with `qint8` inputs/weight and
  `si32` output, wrapped by the output `dequantize → quantize` rescale.
- **QOperator** is imported as `onnx.QLinearConv`, which torch-mlir lowers to the
  same quantized `aten.convolution` form.

`convert-torch-to-linalg` then lowers both to
`linalg.conv_2d_nchw_fchw_q`. The TORQ compiler's `LinalgToTorqHL` conversion
takes that quantized conv and lowers it to `torq_hl.conv2d`.

Not every op follows this exact path. `QuantizeLinear`/`DequantizeLinear`
operations that are not fused into a weighted op are lowered as standalone
`torq_hl` operations by the fallback patterns in `QuantizePattern.cpp`. See
[QuantizePattern.cpp (fallback)](#442-quantizepatterncpp-fallback) for details.

## 2. ONNX Quantization Formats

ORT supports two quantized graph representations.

### QDQ

In **Quantize/Dequantize** mode, ORT inserts `QuantizeLinear` /
`DequantizeLinear` nodes around the original FP32 ops. The original op is
preserved and only its input/output types change:

```
DQ → Conv → Q
DQ → Add → Q
DQ → ReduceMean → Q
```

ORT can quantize virtually any op this way; there is no need for a dedicated
quantized op such as `QLinearReduceMean`.

### QOperator

In **QOperator** mode, ORT replaces supported ops with native int8 ops:

- `Conv` → `QLinearConv`
- `MatMul` / `Gemm` → `QLinearMatMul` / `QGemm`
- `Add` → `QLinearAdd`
- `MaxPool`, `AveragePool`, `GlobalAveragePool`
- `Relu`, `Clip`, `LeakyRelu`, `Sigmoid`, `Softmax`

Unsupported ops are left in FP32 and wrapped with DQ/Q nodes.

### Hybrid

A **hybrid** quantized model mixes QDQ and QOperator representations. When a
full model is quantized with `--quant-format=qoperator`, ORT already produces a
hybrid graph automatically: supported ops become `QLinear*` nodes and
unsupported ops stay as DQ/Q islands. For example:

```
QLinearConv → DequantizeLinear → ReduceMean → Reshape → QuantizeLinear → QGemm
```

TORQ's `--quant-format=hybrid` extends this idea to **per extracted layer**:

- Use **qoperator** when every compute op in the layer is in TORQ's preferred set
  (e.g. `Conv`, `Add`, `MatMul`, `MaxPool`, `Relu`).
- Use **QDQ** otherwise (e.g. `ReduceMean`, `Gemm`).

Because each layer is quantized independently, one layer can be a compact
`QLinearConv` while the next remains a DQ/Q-wrapped `ReduceMean`. On full
ResNet18 this places 47 layers on NSS (`Conv`, `Add`, `Relu`, `MaxPool`,
`Reshape`) and 2 layers on HOST (`ReduceMean`, `Gemm`), with pure qoperator
failing on `Gemm` and pure QDQ missing the more compact qoperator form.

## 3. torq-gen-config Quantization Workflow

`torq-gen-config discover` and `torq-gen-config run` are the two entry points for
quantized ONNX models.

The Python static-quantization helpers accept ONNX Runtime's
`CalibrationDataReader` through `calibration_data_reader`; use it to yield
representative dictionaries keyed by ONNX input name. This lets applications use
ONNX Runtime's existing readers or their own dataset adapters without a Torq
dataset format. Calls without a reader fall back to deterministic synthetic
random calibration inputs.

### Supported Quantization Options

| Option | Effect |
|--------|--------|
| `--quantize` | Enable static int8 quantization for layer tests and the full model. |
| `--per-channel` | Use per-channel weight scales instead of per-tensor scales. |
| `--full-integer` | Rewrite graph inputs/outputs to int8 by stripping the input-side `QuantizeLinear` and output-side `DequantizeLinear`. |
| `--quant-format` | ONNX quantization format:<br>`qdq` — insert `QuantizeLinear`/`DequantizeLinear` nodes around ops (default).<br>`qoperator` — use native int8 ops such as `QLinearConv`.<br>`hybrid` — choose qoperator or QDQ per extracted layer based on op support. |
| `--quant-dtype` | Activation/weight integer dtype. Currently only `A8W8` (signed 8-bit for both, default) is supported. Case-insensitive. |

**Important:** `discover` and `run` must use the same quantization flags. The
compiler JSON is built from the final quantized MLIR; mismatched flags produce a
mismatched executor map.

### Per-Layer vs. Full-Model Quantization

Each extracted layer is quantized **independently** during discovery. A `Relu`
layer becomes its own tiny quantized graph:

```
input → QuantizeLinear → Relu → DequantizeLinear → output
```

So the per-layer test produces a valid reference and a `recommended_executor`
for `Relu` in the report JSON.

When the whole model is quantized, the quantizer can fuse `Relu` into the
preceding `Conv`. In the final quantized MLIR there is no separate `Relu` line,
so the compiler JSON assigns the fused `Conv+Relu` entirely through the `Conv`
entry. This is why you may see a `Relu_...` entry in the report JSON but no
matching line in the compiler JSON.

### CLI Usage

```bash
# Basic quantized discovery (QDQ)
torq-gen-config discover --model model.onnx --skip-mode --quantize

# Full-integer int8 I/O
torq-gen-config discover --model model.onnx --skip-mode --quantize --full-integer

# Quantized, skipping CSS and deduplicating identical layers
torq-gen-config discover \
    --model model.onnx \
    --skip-mode \
    --quantize --full-integer --quant-format=qdq \
    --skip-executors=css --dedup-layers
```

`run` applies the compiler JSON produced by `discover` to the full model. Pass
the same quantization flags so the JSON line numbers match the final quantized
MLIR:

```bash
torq-gen-config run \
    --model model.onnx \
    --output-dir results/ \
    --quantize --full-integer --quant-format=qdq \
    --debug-ir=tmp
```

If the flags do not match, the compiler JSON line numbers will not align with
the full-model MLIR and executor assignment will fail or target the wrong ops.

## 4. Quantized Convolution Zero-Point Correction

Quantized ops with weights need two corrections:

1. **Fold the weight zero-point into the weight tensor.**
2. **Add `-input_zp * sum(adjusted_weight)` as a per-output bias.**

This section derives those corrections for Conv2D. The same structure applies to
any weighted op of the form `y = x * w + b`.

### 4.1 Operation definition

A quantized convolution is typically represented in Linalg as:

```mlir
%out = linalg.conv_2d_nchw_fchw_q {
    dilations = dense<1> : vector<2xi64>,
    strides = dense<2> : vector<2xi64>
} ins(
    %input,       // tensor<...xi8>, quantized activation
    %weight,      // tensor<...xi8>, quantized weight
    %input_zp,    // i32 scalar, input zero-point
    %weight_zp    // i32 scalar, weight zero-point
    : tensor<...xi8>, tensor<...xi8>, i32, i32
) outs(%init : tensor<...xi32>) -> tensor<...xi32>
```

The four input operands are:

| Index | Operand | Meaning |
|-------|---------|---------|
| 0 | `%input` | quantized input activation tensor (`i8`) |
| 1 | `%weight` | quantized weight/filter tensor (`i8`) |
| 2 | `%input_zp` | input zero-point (`i32` scalar) |
| 3 | `%weight_zp` | weight zero-point (`i32` scalar) |

The integer convolution computes, for each output element:

```
conv_out[c, y, x] = sum_{kh,kw,ic} input[y+kh, x+kw, ic] * weight[c, kh, kw, ic]
```

But the mathematically correct result in the integer domain is:

```
correct[c, y, x] = sum_{kh,kw,ic}
    (input[y+kh, x+kw, ic] - input_zp) *
    (weight[c, kh, kw, ic] - weight_zp)
```

The hardware path keeps the input tensor in its as-quantized form, so explicit
dequantization to floating point is avoided. Two corrections are required to
recover `correct` from the raw integer convolution.

### 4.2 Mechanical rewriting

Expand the product inside the summation:

```
(input - input_zp) * (weight - weight_zp)
= input * weight
  - input_zp * weight                    // term A
  - weight_zp * input                    // term B
  + input_zp * weight_zp                 // term C
```

The full sum over the kernel window therefore becomes:

```
correct = conv(input_quantized, weight_quantized)
          - input_zp * sum(weight_quantized)            // term A
          - weight_zp * sum(input_quantized)            // term B
          + input_zp * weight_zp * K                    // term C, K = kernel size
```

Because the integer convolution only computes `conv(input_quantized,
weight_quantized)`, the missing terms must be added explicitly. The terms are
rewritten so that each one is a sub-component that can be lowered independently
by the compiler:

1. `conv(input_quantized, weight_quantized)` is computed by the integer
   convolution hardware.
2. `-input_zp * sum(weight_quantized)` is the input-zero-point correction.
3. `-weight_zp * input` is absorbed by folding `weight_zp` into the weight
   tensor before the convolution.
4. `input_zp * weight_zp * K` is a constant that is already folded into the
   bias (or is zero when `weight_zp == 0`).

After these rewrites the effective computation is:

```
intermediate_i32[c, y, x] = conv_i32(
                                input_quantized,
                                adjusted_weight
                            )
                            + input_zp_correction[c]
                            + bias[c]
```

where:

- `adjusted_weight = weight - weight_zp`
- `input_zp_correction[c] = -input_zp * sum(adjusted_weight[c])`
- `bias[c]` is the optional per-channel bias, typically already in `i32`.

### 4.3 Sub-component definitions

#### 4.3.1 Weight zero-point folding

For every weight element:

```
adjusted_weight[c, kh, kw, ic] = weight[c, kh, kw, ic] - weight_zp
```

This removes term B from the expanded expression because the convolution now
operates on `(weight - weight_zp)`.

**Implementation:** `PatternUtils.cpp` provides `buildWeightWithZp(weights,
weightZp, rewriter)`. It sign-extends the `i8` weight to `i16`, subtracts the
weight zero-point, and materializes an `i16` tensor. For the QConv2D path the
same effect is obtained through `preConversionWeights(...)` in
`Conv2DPattern.cpp`.

**Generalization:** this step is the same for any weighted op. The weights are
simply adjusted by their zero-point before use.

#### 4.3.2 Input zero-point correction

Because the input tensor is not dequantized, term A is missing from the raw
convolution output. We compute it explicitly as a per-channel bias:

```
input_zp_correction[c] = -input_zp * sum_{kh,kw,ic} adjusted_weight[c, kh, kw, ic]
```

This is added to the convolution output before the final rescale.

**Implementation:** `QConv2DPattern.cpp` provides two helpers:

- `computeInputZpCorrection(adjustedWeights, inputZp, rewriter)` — generic
  reduce+multiply path using `linalg.reduce` over all non-output dimensions.
- `buildInputZpCorrectionConstant(weights, inputZp, rewriter)` — constant-only
  fast path that reads the constant weight tensor directly and produces a 1-D
  `i32` constant without creating a `linalg.reduce`.

Both helpers compute the same value; the constant path is used when weights are
constants and the reduce path is the fallback for dynamic weights.

#### 4.3.3 Bias addition

The original QDQ graph may contain a `+ bias` after the convolution. The bias is
added to the corrected integer accumulator:

```
intermediate_i32 = conv_i32(...) + input_zp_correction + bias
```

**Implementation:** `addPerChannelBias(lhs, rhs, rewriter)` in
`QConv2DPattern.cpp` emits a `linalg.generic` that adds two 1-D `i32` tensors.
When both operands are constants (or slices of constants),
`addI32SlicesOfConstants(lhs, rhs, rewriter)` folds them into a single constant
tensor at compile time.

#### 4.3.4 Output rescale and clamp

After the corrected integer accumulation, the result must be rescaled to the
output quantization range. Because the hardware does not perform a floating-point
multiplication at runtime, the FP32 scale is approximated by a fixed-point
multiply-and-shift:

```
output_i8[c, y, x] = saturate(
    round(intermediate_i32[c, y, x] * multiplier / 2^shift) + output_zp
)
```

The scale comes from the surrounding QDQ graph:

```
scale = dequant_scale / quant_scale
      = (input_scale * weight_scale) / output_scale
```

`computeMultiplierAndShift(scale, multiplier, shift)` searches for the largest
multiple-of-4 shift whose rounded multiplier fits in a signed 32-bit integer. If
`scale == 0`, the multiplier is set to `0` and the shift to `28`.

For example, with `scale = 0.5`:

```
shift = 28:  multiplier = round(0.5 * 2^28) = 134217728   (fits in i32)
```

so `scale ~= 134217728 / 2^28 = 0.5`. The actual integer rescale multiplies by
`134217728` and right-shifts by `28` bits.

### 4.4 Lowering each sub-component

#### 4.4.1 QConv2DPattern.cpp

`QConv2dConvert` matches the chain:

```
linalg.conv_2d_nchw_fchw_q -> (optional bias add) -> dequant generic -> quant generic
```

It extracts:

- `%input` from the conv operand 0,
- `%weights` from the conv operand 1,
- `%input_zp` from the conv operand 2,
- `%weight_zp` from the conv operand 3,
- `dequantScale` from the dequant generic,
- `quantScale`, `quantZp`, `quantMin`, `quantMax` from the quant generic.

The rewriter then:

1. Computes `scaleFactor = dequantScale / quantScale` and
   `(multiplier, shift)`.
2. Folds the bias quant generic to a constant `i32` tensor when possible via
   `foldQuantGenericToConstant`.
3. Calls `preConversionWeights(weights, optionalWeightZpV, ...)` to fold
   `weight_zp` into the weights.
4. Computes the input-zero-point correction using the constant fast path when
   possible, otherwise the generic reduce path.
5. Adds the correction into the bias.
6. Folds backward padding so the pad op is fused into the conv.
7. Emits a `torq_hl.conv2d` op with:
   - `input_zp` set to the original input zero-point,
   - `weight_zp` set to `0` because it has already been folded into the weights,
   - `output_zp`, `output_min`, `output_max`, `shift`,
   - an interleaved `scale_bias` tensor of shape `{2 * C}` holding
     `[bias_0, multiplier, bias_1, multiplier, ...]`.

The matching of `dequant -> quant` is intentional: the rewrite only fires when
it can replace the entire QDQ sandwich with a single hardware op. If the chain
is not matched, the fallback patterns in `QuantizePattern.cpp` handle the
standalone quant and dequant generics.

#### 4.4.2 QuantizePattern.cpp (fallback)

When a standalone quantize or dequantize generic survives (for example, because
folding into a weighted op failed), `QuantizePattern.cpp` lowers them to
`torq_hl` operations.

##### Quantize generic

A quantize generic computes:

```
out_i8 = clamp(round(input / scale) + zp, min, max)
```

The pattern `QuantizeOpConversion` rewrites it as:

1. `torq_hl.act` `f2f` (if input is f32, cast to bf16),
2. `torq_hl.mul` with multiplier `1/scale` and bias `zp` in bf16,
3. `torq_hl.act` `f2i` (bf16 -> i32),
4. `torq_hl.fma` with identity weights/bias to clamp to `[min, max]` and cast to
   `i8`.

This is a fallback and is not the preferred path for quant ops that can be
folded into `torq_hl.conv2d`.

##### Dequantize generic

A dequantize generic computes:

```
out = (input_i8 - zp) * scale
```

The pattern `DequantizeOpConversion` rewrites it as:

1. `torq_hl.act` `i2f` (i8 -> bf16),
2. `torq_hl.mul` with multiplier `scale` in bf16,
3. `torq_hl.act` `f2f` (bf16 -> f32) if the original output type is f32.

Again, this is a fallback for cases where the dequant cannot be fused with a
preceding quantized op.

### 4.5 Special cases

#### 4.5.1 `weight_zp == 0`

When the weight zero-point is zero, term B and term C of the expansion both
vanish. The correction simplifies to:

```
input_zp_correction[c] = -input_zp * sum(weight[c])
```

and the weight tensor can be used without adjustment. This is the common case
for ONNX Runtime static quantization, where symmetric weight quantization is
used.

#### 4.5.2 `input_zp == 0`

When the input zero-point is zero, term A vanishes. The input-zero-point
correction becomes a zero tensor and can be skipped entirely. The effective
computation reduces to:

```
intermediate_i32 = conv_i32(input, adjusted_weight) + bias
```

#### 4.5.3 Both zero-points are zero

Both corrections vanish and the integer convolution directly computes the
correct result:

```
intermediate_i32 = conv_i32(input, weight) + bias
```

#### 4.5.4 Dynamic weights

If the weights are not compile-time constants, `buildInputZpCorrectionConstant`
returns `nullptr` and the pattern falls back to `computeInputZpCorrection`,
which emits a `linalg.reduce` followed by a `linalg.generic` multiply. The
result is then added to the bias with `addPerChannelBias`. Later constant-folding
passes can still simplify the graph if the weights become constant later in the
pipeline.

### 4.6 Why the other expanded terms disappear

Recall the full expansion:

```
(input - input_zp) * (weight - weight_zp)
= input * weight
  - input_zp * weight
  - weight_zp * input
  + input_zp * weight_zp
```

- `input * weight` is the raw integer convolution output.
- `-input_zp * weight` is the input-zero-point correction.
- `-weight_zp * input` becomes zero after folding the weight zero-point into the
  weights.
- `+input_zp * weight_zp * K` is a constant folded into the bias or omitted when
  `weight_zp == 0`.

### 4.7 Generalization to other weighted ops

The same two corrections apply to any quantized op with weights:

| Op | Weight zero-point | Input zero-point correction |
|----|-------------------|----------------------------|
| Conv2D | `adjusted_w = w - weight_zp` | `-input_zp * sum_spatial_and_ic(adjusted_w[c])` |
| Conv1D | `adjusted_w = w - weight_zp` | `-input_zp * sum_kw_and_ic(adjusted_w[c])` |
| Conv3D | `adjusted_w = w - weight_zp` | `-input_zp * sum_spatial_and_ic(adjusted_w[c])` |
| DepthwiseConv2D | `adjusted_w = w - weight_zp` | `-input_zp * sum_spatial(adjusted_w[c])` |
| MatMul / FC | `adjusted_w = w - weight_zp` | `-input_zp * sum_input_dim(adjusted_w[out, in])` |

The only difference is which dimensions are summed to compute the
input-zero-point correction. The principle remains:

1. **Fold `weight_zp` into the weight tensor.**
2. **Add `-input_zp * sum(adjusted_weight)` as a per-output bias.**

## 5. References

### ONNX Runtime quantization

- ONNX Runtime Quantization docs: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

### Quantization theory and best-practice papers

- Wu et al., "Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation", 2020.
  - arXiv: https://arxiv.org/abs/2004.09602
  - PDF: https://arxiv.org/pdf/2004.09602
- Nagel et al., "A White Paper on Neural Network Quantization", 2021.
  - https://arxiv.org/abs/2106.08295
- Krishnamoorthi, "Quantizing Deep Convolutional Networks for Efficient Inference: A Whitepaper", 2018.
  - https://arxiv.org/abs/1806.08342
- Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference", CVPR 2018.
  - https://arxiv.org/abs/1712.05877
