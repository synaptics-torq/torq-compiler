// Batch-invariant [K,N] weight broadcast over batch (the FFN up-projection form). Weight
// and output each exceed LRAM, so parallel tiling must cut N — the only dim that shrinks
// the weight. Tolerances allow bf16 K-chunked partials on smaller-LRAM configs.
// TORQ_FP_MAX_TOL: 0.05
// TORQ_USE_ABS_TOL_GATE: 1
// TORQ_FP_ABS_TOL_FRAC: 0.02
module {
  func.func @main(%arg0: tensor<100x1x512xbf16>, %arg1: tensor<512x3200xbf16>) -> (tensor<100x1x3200xbf16>) {
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<100x1x3200xbf16>
    %init = linalg.fill ins(%cst : bf16) outs(%0 : tensor<100x1x3200xbf16>) -> tensor<100x1x3200xbf16>
    %bcast_init = tensor.empty() : tensor<100x512x3200xbf16>
    %bcast = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%arg1 : tensor<512x3200xbf16>) outs(%bcast_init : tensor<100x512x3200xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<100x512x3200xbf16>
    %result = linalg.batch_matmul ins(%arg0, %bcast : tensor<100x1x512xbf16>, tensor<100x512x3200xbf16>) outs(%init : tensor<100x1x3200xbf16>) -> tensor<100x1x3200xbf16>
    return %result : tensor<100x1x3200xbf16>
  }
}
