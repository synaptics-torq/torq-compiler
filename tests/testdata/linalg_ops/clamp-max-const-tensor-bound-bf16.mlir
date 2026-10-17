// One-sided clamp (max toward a bound), bound as a 0-d constant tensor input
// on the *second* input (data first) -- the other operand order of the torch
// ONNX HardSigmoid decomposition. Covers ClampInfo.dataInputIdx = 0 in
// matchOneSidedClampConstInput (the min test above covers idx = 1).
module {
  func.func public @main(%arg0: tensor<1x1024x64xbf16>) -> tensor<1x1024x64xbf16> {
    %lo = arith.constant dense<0.000000e+00> : tensor<1xbf16>
    %e = tensor.empty() : tensor<1x1024x64xbf16>
    %r = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%arg0, %lo : tensor<1x1024x64xbf16>, tensor<1xbf16>) outs(%e : tensor<1x1024x64xbf16>) {
    ^bb0(%in: bf16, %b: bf16, %o: bf16):
      %c = arith.cmpf ogt, %in, %b : bf16
      %s = arith.select %c, %in, %b : bf16
      linalg.yield %s : bf16
    } -> tensor<1x1024x64xbf16>
    return %r : tensor<1x1024x64xbf16>
  }
}
