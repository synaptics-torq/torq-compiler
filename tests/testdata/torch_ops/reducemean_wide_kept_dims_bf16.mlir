// TORQ_FP_MAX_TOL: 0.06
// TORQ_FP_AVG_TOL: 0.02
// Regression: a reduction whose KEPT (parallel) dimensions alone exceed LRAM.
//
// PP-OCRv6-tiny rec_w2432 reduces [1,160,3,608] bf16 over the size-3 H axis;
// this test doubles the channels twice ([1,640,3,608], 2,334,720 B) so the
// kept dimensions exceed the usable LRAM (509,952 B) even when slicing halves
// the channel dim across the two slices. No reduction-dim tile can fix that:
// even at reduction tile 1 the full-width input slice plus the untiled output
// do not fit. TileReductionForLram only halved the reduction dimension, so it
// emitted
//   "failed to tile reduction dimension to fit LRAM size 509952 bytes"
// and, because that hard error also fires inside tile-and-fuse's memory-check
// simulation pipeline, it poisoned the fit search of every group containing
// such a reduction (checkTileFitsInMemory reports failure instead of
// "does not fit"), so the group was left untiled and the final pipeline
// failed on the same error.
//
// The fix adds a fallback that also halves the largest (parallel) dimensions
// until the estimated operand bytes fit, keeping the reduction whole when
// possible for a single-pass accumulation.
//
// Before the fix: torq-compile fails with the error above. After: compiles
// and matches the reference.
module {
  func.func @main(%arg0: !torch.vtensor<[1,640,3,608],bf16>) -> !torch.vtensor<[1,640,608],bf16> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "2.9.1+cu128"} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<2> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64>
    %1 = torch.operator "onnx.ReduceMean"(%arg0, %0) {torch.onnx.keepdims = 0 : si64, torch.onnx.noop_with_empty_axes = 0 : si64} : (!torch.vtensor<[1,640,3,608],bf16>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[1,640,608],bf16>
    return %1 : !torch.vtensor<[1,640,608],bf16>
  }
}
