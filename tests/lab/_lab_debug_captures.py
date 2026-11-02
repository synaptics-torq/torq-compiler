# Copyright 2026 Synaptics Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Captured compiler stderr for classifier tests.

Keep diagnostic lines exact. Replace machine paths with <<SRC>> and
<<BUILD>>, and mark omitted blocks as "... (trimmed) ...". Each entry
records its model, flags, and build; label synthetic output explicitly.
"""

from typing import Dict

CAPTURES: Dict[str, str] = {
    # One-op f32 Erf (opset 13), both fallbacks disabled; dev build, diag flags.
    # Command: torq-lab compile erf_f32.onnx --compiler-option=--torq-disable-css
    #          --compiler-option=--torq-disable-host
    "no-npu-lowering-erf-both-disabled.dev.stderr": """
Compile failed: Error invoking torq-compile (code 1)
Diagnostics:
<<SRC>>/work/erf_f32.mlir:4:10: error: 'linalg.generic' op unsupported in function body
    %0 = torch.operator "onnx.Erf"(%arg0) : (!torch.vtensor<[1,64],f32>) -> !torch.vtensor<[1,64],f32>
         ^
  ... (trimmed) ...
<<SRC>>/work/erf_f32.mlir:2:3: error: failed to serialize to torq binary
<<SRC>>/work/erf_f32.mlir:2:3: error: failed to serialize executable for target backend torq
<<SRC>>/work/erf_f32.mlir:2:3: error: failed to serialize executables
Command: torq-compile <<SRC>>/work/erf_f32.mlir -o <<SRC>>/work/erf_f32.vmfb --torq-hw=SL2610 --torq-target-host-triple=native --torq-disable-css --torq-disable-host
""".lstrip(),
    # One-op int8 Div, executor map pins 5:10 to css, CSS disabled; dev build.
    # The first error line carries the tile-and-fuse probe suffix.
    "no-npu-lowering-forced-css-disabled.dev.stderr": """
Compile failed: Error invoking torq-compile (code 1)
Diagnostics:
<<SRC>>/work/div_int8.mlir:5:10: error: Cannot execute operation on CSS as disabled by command line option (encountered while running the pipeline to checking if a tile fits in memory)
    %1 = torch.operator "onnx.Div"(%arg0, %0) : (!torch.vtensor<[1,64],si8>, !torch.vtensor<[1,64],si8>) -> !torch.vtensor<[1,64],si8>
         ^
  ... (trimmed) ...
<<SRC>>/work/div_int8.mlir:5:10: error: Cannot execute operation on CSS as disabled by command line option
    %1 = torch.operator "onnx.Div"(%arg0, %0) : (!torch.vtensor<[1,64],si8>, !torch.vtensor<[1,64],si8>) -> !torch.vtensor<[1,64],si8>
         ^
  ... (trimmed) ...
<<SRC>>/work/div_int8.mlir:2:3: error: failed to run translation of source executable to target executable for backend #hal.executable.target<"torq", "torq-fb">
Command: torq-compile <<SRC>>/work/div_int8.mlir -o <<SRC>>/work/div_int8.vmfb --torq-hw=SL2610 --torq-target-host-triple=native --torq-disable-css --torq-executor-map=<map: 5:10 -> css>
""".lstrip(),
    # One-op f32 Erf, executor map pins 4:10 to host, Host disabled; dev build.
    "no-npu-lowering-forced-host-disabled.dev.stderr": """
Compile failed: Error invoking torq-compile (code 1)
Diagnostics:
<<SRC>>/work/erf_f32.mlir:4:10: error: Cannot execute operation on Host as disabled by command line option
    %0 = torch.operator "onnx.Erf"(%arg0) : (!torch.vtensor<[1,64],f32>) -> !torch.vtensor<[1,64],f32>
         ^
  ... (trimmed) ...
<<SRC>>/work/erf_f32.mlir:2:3: error: failed to run translation of source executable to target executable for backend #hal.executable.target<"torq", "torq-fb">
Command: torq-compile <<SRC>>/work/erf_f32.mlir -o <<SRC>>/work/erf_f32.vmfb --torq-hw=SL2610 --torq-target-host-triple=native --torq-disable-host --torq-executor-map=<map: 4:10 -> host>
""".lstrip(),
    # In-tree stablehlo testdata (262144x640xbf16 gather); plain compile, dev build.
    "memory-stablehlo-gather.dev.stderr": """
Compile failed: Error invoking torq-compile (code 1)
Diagnostics:
  ... (trimmed) ...
<<SRC>>/tests/testdata/stablehlo_ops/gather_262144_640.mlir:3:10: error: failed to allocate LRAM addresses
    %7 = "stablehlo.gather"(%arg0, %6) <{dimension_numbers = #stablehlo.gather<offset_dims = [2], collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 2>, indices_are_sorted = false, slice_sizes = array<i64: 1, 640>}> : (tensor<262144x640xbf16>, tensor<1x256x1xi32>) -> tensor<1x256x640xbf16>
         ^
  ... (trimmed) ...
<<SRC>>/tests/testdata/stablehlo_ops/gather_262144_640.mlir:3:10: error: failed to run translation of source executable to target executable for backend #hal.executable.target<"torq", "torq-fb">
Command: torq-compile <<SRC>>/tests/testdata/stablehlo_ops/gather_262144_640.mlir -o <<SRC>>/work/gather_262144_640.vmfb --torq-hw=SL2610 --torq-target-host-triple=native
""".lstrip(),
    # Oversized bf16 MatMul [1,2048]x[2048,2048]; the LRAM OOM surfaces as a
    # 'tensor.empty' verifier error (a known compiler bug, filed separately).
    "memory-oversized-matmul-tensor-empty.dev.stderr": """
Compile failed: Error invoking torq-compile (code 1)
Diagnostics:
  ... (trimmed) ...
// -----// IR Dump After AssignLramAddresses Failed (torq-assign-lram-addresses) //----- //
  ... (trimmed) ...
<<SRC>>/work/matmul_big_bf16.mlir:5:10: error: 'tensor.empty' op incorrect number of dynamic sizes, has 0, expected 2
    %1 = torch.operator "onnx.MatMul"(%arg0, %0) : (!torch.vtensor<[1,2048],bf16>, !torch.vtensor<[2048,2048],bf16>) -> !torch.vtensor<[2048,2048],bf16>
         ^
Command: torq-compile <<SRC>>/work/matmul_big_bf16.mlir -o <<SRC>>/work/matmul_big_bf16.vmfb --torq-hw=SL2610 --torq-target-host-triple=native
""".lstrip(),
    # f32 Conv2D (stride 1); the NSS conv matcher admits the dtype, the kernel
    # asserts. dev build; rc -6.
    "assert-conv-f32-iwidth.dev.stderr": """
Compile failed: Error invoking torq-compile (code -6)
Diagnostics:
torq-compile: <<SRC>>/compiler/torq/Utils/Kernel.cpp:3604: int mlir::syna::torq::Alu::iWidth(DType, DType, int) const: Assertion `iType == DType::bf16 && wType == DType::bf16' failed.
Please report issues to https://github.com/iree-org/iree/issues and include the crash backtrace.
Command: torq-compile <<SRC>>/work/conv_stride1_f32.mlir -o <<SRC>>/work/conv_stride1_f32.vmfb --torq-hw=SL2610 --torq-target-host-triple=native
""".lstrip(),
    # fp16 Conv2D (stride 1); the kernel asserts on the element type. dev build.
    "assert-conv-fp16-dtype.dev.stderr": """
Compile failed: Error invoking torq-compile (code -6)
Diagnostics:
torq-compile: <<SRC>>/compiler/torq/Utils/Kernel.cpp:880: mlir::syna::torq::LData::LData(const MemRefType &): Assertion `elementType != DType::none && "Invalid element type"' failed.
Please report issues to https://github.com/iree-org/iree/issues and include the crash backtrace.
Command: torq-compile <<SRC>>/work/conv_stride1_fp16.mlir -o <<SRC>>/work/conv_stride1_fp16.vmfb --torq-hw=SL2610 --torq-target-host-triple=native
""".lstrip(),
    # fp32 one-op MatMul, executor map pins 5:10 to nss, both fallbacks
    # disabled; dev build. Kernel.cpp line drifts between builds (see release).
    "assert-matmul-nss-fp32.dev.stderr": """
Compile failed: Error invoking torq-compile (code -6)
Diagnostics:
Invalid weight conversion from fp32 to bf16
torq-compile: <<SRC>>/compiler/torq/Utils/Kernel.cpp:3353: WeightFormat mlir::syna::torq::getWeightMemoryFormat(DType, DType): Assertion `false && "Invalid weight memory format conversion"' failed.
Please report issues to https://github.com/iree-org/iree/issues and include the crash backtrace.
Command: torq-compile <<SRC>>/work/matmul_assert_f32.mlir -o <<SRC>>/work/matmul_assert_f32.vmfb --torq-hw=SL2610 --torq-target-host-triple=native --torq-disable-css --torq-disable-host --torq-executor-map=<map: 5:10 -> nss>
""".lstrip(),
    # Same model and flags plus --mlir-pass-pipeline-crash-reproducer=crash-repro.mlir;
    # dev build. The reproducer flag flips the abort into a reported pipeline
    # failure (rc 1); the assert text still precedes it.
    "assert-matmul-nss-fp32-with-crash-reproducer.dev.stderr": """
Compile failed: Error invoking torq-compile (code 1)
Diagnostics:
  ... (trimmed) ...
Invalid weight conversion from fp32 to bf16
torq-compile: <<SRC>>/compiler/torq/Utils/Kernel.cpp:3353: WeightFormat mlir::syna::torq::getWeightMemoryFormat(DType, DType): Assertion `false && "Invalid weight memory format conversion"' failed.
  ... (trimmed) ...
<<SRC>>/work/matmul_assert_f32.mlir:2:3: error: Failures have been detected while processing an MLIR pass pipeline
<<SRC>>/work/matmul_assert_f32.mlir:2:3: note: Pipeline failed while executing [`ConvertSliceProgramToTorqHw` on 'torq_hl.program' operation]: reproducer generated at `crash-repro.mlir`
// -----// IR Dump After TORQLowerExecutableTarget Failed (iree-torq-lower-executable-target-pass) //----- //
module {
  ... (trimmed) ...
Command: torq-compile <<SRC>>/work/matmul_assert_f32.mlir -o <<SRC>>/work/matmul_assert_f32.vmfb --torq-hw=SL2610 --torq-target-host-triple=native --torq-disable-css --torq-disable-host --torq-executor-map=<map: 5:10 -> nss> --mlir-pass-pipeline-crash-reproducer=crash-repro.mlir
""".lstrip(),
    # One-op Erf with a dynamic input dim [1,"N"]; fails in torq.lab before any
    # compiler run (LabError text, not compiler stderr).
    "dynamic-shape-erf.stderr": """
Compile failed: the TORQ compiler does not support dynamic shapes: erf_dynamic.onnx has dynamic dimension(s) in its entry function I/O: !torch.vtensor<[1,?],f32>

Re-export the model with static shapes and try again.
""".lstrip(),
    # One-op LpPool; torch-mlir's ONNX lowering leaves the op marked illegal;
    # dev build, diag flags.
    "frontend-unsupported-op-lppool.dev.stderr": """
Compile failed: Error invoking torq-compile (code 1)
Diagnostics:
<<SRC>>/work/lppool_f32.mlir:4:10: error: failed to legalize operation 'torch.operator' that was explicitly marked illegal
    %0 = torch.operator "onnx.LpPool"(%arg0) {torch.onnx.kernel_shape = [3 : si64, 3 : si64], torch.onnx.p = 2 : si64} : (!torch.vtensor<[1,1,16,16],f32>) -> !torch.vtensor<[1,1,14,14],f32>
         ^
// -----// IR Dump After ConvertTorchOnnxToTorch Failed (convert-torch-onnx-to-torch) //----- //
Command: torq-compile <<SRC>>/work/lppool_f32.mlir -o <<SRC>>/work/lppool_f32.vmfb --torq-hw=SL2610 --torq-target-host-triple=native --mlir-print-ir-after-failure --mlir-print-debuginfo --mlir-elide-elementsattrs-if-larger=16 --torq-show-pass-progress --mlir-disable-threading
""".lstrip(),
    # Corrupted ONNX (node input 'ghost' undefined); onnx.checker at import.
    "invalid-model-erf-corrupt.stderr": """
Compile failed: failed to import ONNX <<SRC>>/models/erf_corrupt.onnx: ONNX model validation failed for <<SRC>>/models/erf_corrupt.onnx
Error: Nodes in a graph must be topologically sorted, however input 'ghost' of node:
name: erf OpType: Erf
 is not output of any previous nodes.
""".lstrip(),
    # Synthetic: import_onnx subprocess non-zero. Shape follows
    # convert_onnx_to_mlir's RuntimeError; no input currently triggers it.
    "import-failed-canned.stderr": """
Compile failed: failed to import ONNX <<SRC>>/models/broken.onnx: iree.compiler.tools.import_onnx failed for <<SRC>>/models/broken.onnx
Return code: 1
stdout:
(empty)
stderr:
onnx_importer.OnnxImportError: Unhandled ONNX attribute type code 11: <onnx.AttributeProto proto>
Model file size: 2048 bytes
""".lstrip(),
    # Missing CSS config with --torq-show-pass-progress: the progress log starts
    # during setup, so its last unmatched '+' (TileAndFuse) is noise; the
    # css-config line (no trailing newline) is the excerpt.
    "environment-missing-css-config-progressive.dev.stderr": """
[TORQ erf_dispatch_0_elementwise_1x64_f32] + MarkPatternsForTileAndFuse
[TORQ erf_dispatch_0_elementwise_1x64_f32] - MarkPatternsForTileAndFuse (0 ms, +192 kB)
[TORQ erf_dispatch_0_elementwise_1x64_f32] + ReductionTiling
[TORQ erf_dispatch_0_elementwise_1x64_f32] - ReductionTiling (0 ms, 0 MB)
[TORQ erf_dispatch_0_elementwise_1x64_f32] + TileAndFuse
Unable to find css config no_such_css_config""",
    # Synthetic: tool timeout (ToolError with rc None; requires --timeout N).
    "timeout-canned.stderr": """
Compile failed: Error invoking torq-compile (timeout)
Diagnostics:
timed out after 300s
Command: torq-compile <<SRC>>/work/model.mlir -o <<SRC>>/work/model.vmfb --torq-hw=SL2610 --torq-target-host-triple=native
""".lstrip(),
    # Stale in-tree tosa testdata: the dialect no longer knows the op.
    "parse-error-stale-tosa-fc.stderr": """
Compile failed: Unable to parse module assembly:
error: "-":5:12: custom op 'tosa.fully_connected' is unknown
""".lstrip(),
    # Stale in-tree tosa testdata: the op's operand list changed.
    "parse-error-stale-tosa-dwconv.stderr": """
Compile failed: Unable to parse module assembly:
error: "-":5:10: 'tosa.depthwise_conv2d' op expected 5 operands, but found 3
 note: "-":5:10: see current operation: %2 = "tosa.depthwise_conv2d"(%arg0, %1, %0) <{dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>}> {quantization_info = #tosa.conv_quant<input_zp = 0, weight_zp = 0>} : (tensor<1x32x32x21xi16>, tensor<32x32x21x1xi8>, tensor<21xi32>) -> tensor<1x1x1x21xi32>
""".lstrip(),
    # Stale in-tree tosa testdata: a verifier violation (a one-off compiler
    # bug with its own issue: pinned unknown, not invalid-source).
    "verify-error-stale-tosa-collapse-shape.stderr": """
Compile failed: Error invoking torq-compile (code 1)
Diagnostics:
<<SRC>>/extras/tests/testdata/tosa_ops/conv2d-matmul-wzp.mlir:7:10: error: 'tensor.collapse_shape' op expected dimension 0 of collapsed type to be static value of 256
    %0 = tosa.conv2d %arg0, %cst_0, %cst, %cst_1, %cst_2 {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<1x16x16x32xi8>, tensor<8x1x1x32xi8>, tensor<8xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x16x16x8xi32>
         ^
  ... (trimmed) ...
<<SRC>>/extras/tests/testdata/tosa_ops/conv2d-matmul-wzp.mlir:2:3: error: failed to run translation of source executable to target executable for backend #hal.executable.target<"torq", "torq-fb">
Command: torq-compile <<SRC>>/extras/tests/testdata/tosa_ops/conv2d-matmul-wzp.mlir -o <<SRC>>/work/conv2d-matmul-wzp.vmfb --torq-hw=SL2610 --torq-target-host-triple=native --torq-disable-host --torq-disable-css
""".lstrip(),
    # One-op int8 Reciprocal, CSS disabled: the conversion error and the
    # assert it triggers share the stream; the assert rule must win.
    "input-conversion-reciprocal-i8-assert.dev.stderr": """
Compile failed: Error invoking torq-compile (code -6)
Diagnostics:
<unknown>:0: error: expected floating point type
torq-compile: <<SRC>>/third_party/iree/third_party/llvm-project/mlir/include/mlir/IR/StorageUniquerSupport.h:180: static ConcreteT mlir::detail::StorageUserBase<mlir::FloatAttr, mlir::Attribute, mlir::detail::FloatAttrStorage, mlir::detail::AttributeUniquer, mlir::TypedAttr::Trait>::get(MLIRContext *, Args &&...) [ConcreteT = mlir::FloatAttr, BaseT = mlir::Attribute, StorageT = mlir::detail::FloatAttrStorage, UniquerT = mlir::detail::AttributeUniquer, Traits = <mlir::TypedAttr::Trait>, Args = <mlir::Type &, llvm::APFloat>]: Assertion `succeeded( ConcreteT::verifyInvariants(getDefaultDiagnosticEmitFn(ctx), args...))' failed.
Please report issues to https://github.com/iree-org/iree/issues and include the crash backtrace.
Command: torq-compile <<SRC>>/work/reciprocal_int8.mlir -o <<SRC>>/work/reciprocal_int8.vmfb --torq-hw=SL2610 --torq-target-host-triple=native --torq-disable-css
""".lstrip(),
    # Synthetic release-shape twin of the capture above: what an
    # assertion-compiled-out build shows (diagnostic only, rc 1, no assert).
    "input-conversion-expected-float-type.release.stderr": """
Compile failed: Error invoking torq-compile (code 1)
Diagnostics:
<unknown>:0: error: expected floating point type
Command: torq-compile <<SRC>>/work/reciprocal_int8.mlir -o <<SRC>>/work/reciprocal_int8.vmfb --torq-hw=SL2610 --torq-target-host-triple=native --torq-disable-css
""".lstrip(),
    # Synthetic (verified live in QA s17): TFLite import with the [tflite]
    # extra missing: tosa-converter-for-tflite is not on PATH. Shape follows
    # convert_tflite_to_mlir's RuntimeError wrapped by _import_to_mlir.
    "tflite-import-missing-tool.stderr": """
Compile failed: failed to import TFLite <<SRC>>/models/tiny.tflite: tosa-converter-for-tflite not found on PATH; install the [tflite] extra
""".lstrip(),
    # Synthetic (verified live in QA s17): tosa-converter-for-tflite exited
    # non-zero. Shape follows convert_tflite_to_mlir's RuntimeError.
    "tflite-import-converter-failed.stderr": """
Compile failed: failed to import TFLite <<SRC>>/models/tiny.tflite: tosa-converter-for-tflite failed for <<SRC>>/models/tiny.tflite
Return code: 1
stdout:
(empty)
stderr:
tosa-converter-for-tflite: tiny.tflite: unsupported op
""".lstrip(),
}
