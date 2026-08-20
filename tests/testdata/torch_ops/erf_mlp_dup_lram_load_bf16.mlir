// TORQ_FP_MAX_TOL: 0.2
// TORQ_FP_AVG_TOL: 0.03
// TORQ_USE_ABS_TOL_GATE: 1
// TORQ_FP_ABS_TOL_FRAC: 0.001
// Regression: back-to-back LRAM interval double-count rejects a required
// duplicate-load merge.
//
// An MLP block of PP-OCRv6-tiny rec_w2432: 1x1 Conv 96->192, exact-erf GELU
// (Div/Erf/Add/Mul/Mul) at [1,192,3,608] bf16, and a 1x1 Conv 192->160.
// The GELU lowering emits an elementwise square mul(v, v);
// tile-and-fuse tiles it into chunks whose two operands are identical
// extract_slices, and bufferization stages each operand as its own
// torq_hl.load into a separate LRAM buffer. A chunk then needs three
// 233,472 B buffers plus code (705,544 B) which can never fit the 523,008 B
// of usable LRAM; with the two loads merged it needs 472,072 B and fits.
//
// EliminateRedundantLramLoads finds the merge, but filterByLramSize rejected
// it: computePeakLramUsage sorted its interval sweep events by op index only,
// so at equal indices a begin (+size) could be counted before the end (-size)
// of a back-to-back buffer, double-counting both. Whether that inflation
// exceeds the ceiling depends on the sort's input order, which follows
// DenseMap pointer iteration order, so the failure is nondeterministic
// run-to-run (this file fails the unfixed compiler on most runs):
//   "cannot allocate op (pinned 472072 B + required 233472 B):
//    exceeds 523008 B usable (capacity)"
//
// The fix orders frees before allocations at equal sweep indices (making the
// estimate correct and deterministic) and accepts outright any reuse whose
// dropped interval is contained in the kept one, which provably cannot raise
// the peak.
//
// Before the fix: torq-compile usually fails with the error above. After:
// compiles deterministically and matches the reference. The abs-tol gate
// ignores near-zero cancellation outputs (|expected| < ~0.05 on a +-118
// range), whose bf16 ULP-sized absolute error trips the relative metric.
// Weights are bf16-exact splat constants rather than the trained values: the
// failure is a compile-time LRAM layout property, so weight values are
// irrelevant, and splats keep this file ~5 KB instead of 203 KB. Regenerating
// the graph through torch.export does NOT reproduce the bug (0/10 compiles vs
// 9/10 for this file): the sweep double-count is sensitive to the exact
// program layout the ONNX importer produces.
module {
  func.func @"Extracted from {PaddlePaddle Graph in PIR mode}"(%arg0: !torch.vtensor<[1,96,3,608],bf16>) -> !torch.vtensor<[1,160,3,608],bf16> attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "onnx.utils.extract_model", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1.414062e+00> : tensor<bf16>} : () -> !torch.vtensor<[],bf16> 
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1.000000e+00> : tensor<bf16>} : () -> !torch.vtensor<[],bf16> 
    %2 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<5.000000e-01> : tensor<bf16>} : () -> !torch.vtensor<[],bf16> 
    %3 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<6.250000e-02> : tensor<192x96x1x1xbf16>} : () -> !torch.vtensor<[192,96,1,1],bf16> 
    %4 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1.250000e-01> : tensor<192xbf16>} : () -> !torch.vtensor<[192],bf16> 
    %5 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<-3.125000e-02> : tensor<160x192x1x1xbf16>} : () -> !torch.vtensor<[160,192,1,1],bf16> 
    %6 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<9.375000e-02> : tensor<160xbf16>} : () -> !torch.vtensor<[160],bf16> 
    %none = torch.constant.none
    %7 = torch.operator "onnx.Conv"(%arg0, %3, %4) {torch.onnx.dilations = [1 : si64, 1 : si64], torch.onnx.group = 1 : si64, torch.onnx.kernel_shape = [1 : si64, 1 : si64], torch.onnx.pads = [0 : si64, 0 : si64, 0 : si64, 0 : si64], torch.onnx.strides = [1 : si64, 1 : si64]} : (!torch.vtensor<[1,96,3,608],bf16>, !torch.vtensor<[192,96,1,1],bf16>, !torch.vtensor<[192],bf16>) -> !torch.vtensor<[1,192,3,608],bf16> 
    %8 = torch.operator "onnx.Div"(%7, %0) : (!torch.vtensor<[1,192,3,608],bf16>, !torch.vtensor<[],bf16>) -> !torch.vtensor<[1,192,3,608],bf16> 
    %9 = torch.operator "onnx.Erf"(%8) : (!torch.vtensor<[1,192,3,608],bf16>) -> !torch.vtensor<[1,192,3,608],bf16> 
    %10 = torch.operator "onnx.Add"(%9, %1) : (!torch.vtensor<[1,192,3,608],bf16>, !torch.vtensor<[],bf16>) -> !torch.vtensor<[1,192,3,608],bf16> 
    %11 = torch.operator "onnx.Mul"(%7, %10) : (!torch.vtensor<[1,192,3,608],bf16>, !torch.vtensor<[1,192,3,608],bf16>) -> !torch.vtensor<[1,192,3,608],bf16> 
    %12 = torch.operator "onnx.Mul"(%11, %2) : (!torch.vtensor<[1,192,3,608],bf16>, !torch.vtensor<[],bf16>) -> !torch.vtensor<[1,192,3,608],bf16> 
    %13 = torch.operator "onnx.Conv"(%12, %5, %6) {torch.onnx.dilations = [1 : si64, 1 : si64], torch.onnx.group = 1 : si64, torch.onnx.kernel_shape = [1 : si64, 1 : si64], torch.onnx.pads = [0 : si64, 0 : si64, 0 : si64, 0 : si64], torch.onnx.strides = [1 : si64, 1 : si64]} : (!torch.vtensor<[1,192,3,608],bf16>, !torch.vtensor<[160,192,1,1],bf16>, !torch.vtensor<[160],bf16>) -> !torch.vtensor<[1,160,3,608],bf16> 
    return %13 : !torch.vtensor<[1,160,3,608],bf16>
  }
}
