module {
  func.func public @main(%arg0: tensor<1x1024x64xbf16>) -> tensor<1x1024x64xbf16> {
    %cst = arith.constant 1.0 : bf16
    %0 = tensor.empty() : tensor<1x1024x64xbf16>
    %1 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x1024x64xbf16>) outs(%0 : tensor<1x1024x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      // TORQ_FP_MAX_TOL: 0.015267175572519083373568271611
      %2 = math.exp %in : bf16
      linalg.yield %2 : bf16
    } -> tensor<1x1024x64xbf16>
    return %1 : tensor<1x1024x64xbf16>
  }
}
