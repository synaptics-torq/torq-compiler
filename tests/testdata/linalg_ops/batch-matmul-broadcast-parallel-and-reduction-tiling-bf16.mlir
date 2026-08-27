// A batch_matmul needing BOTH tilings: batch-invariant LHS broadcast (the collapse
// rewrite needs a batch-invariant RHS), an all-parallel output (512 KiB) over the LRAM
// budget so reduction tiling must skip the untiled op, and K-operands that put any
// useful parallel tile over budget so each parallel tile is then reduction-tiled.
// Tolerances: K-chunked bf16 accumulation carries partials in bf16; gate near-zero
// cancellation at 2% of dynamic range, 5% relative elsewhere.
// TORQ_FP_MAX_TOL: 0.05
// TORQ_USE_ABS_TOL_GATE: 1
// TORQ_FP_ABS_TOL_FRAC: 0.02
module {
  func.func @main(%arg0: tensor<128x2048xbf16>, %arg1: tensor<16x2048x128xbf16>) -> (tensor<16x128x128xbf16>) {
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<16x128x128xbf16>
    %init = linalg.fill ins(%cst : bf16) outs(%0 : tensor<16x128x128xbf16>) -> tensor<16x128x128xbf16>
    %bcast_init = tensor.empty() : tensor<16x128x2048xbf16>
    %bcast = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%arg0 : tensor<128x2048xbf16>) outs(%bcast_init : tensor<16x128x2048xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<16x128x2048xbf16>
    %result = linalg.batch_matmul ins(%bcast, %arg1 : tensor<16x128x2048xbf16>, tensor<16x2048x128xbf16>) outs(%init : tensor<16x128x128xbf16>) -> tensor<16x128x128xbf16>
    return %result : tensor<16x128x128xbf16>
  }
}
