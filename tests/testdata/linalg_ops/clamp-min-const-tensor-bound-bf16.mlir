// One-sided clamp (min toward a bound) where the bound arrives as a 0-d
// constant *tensor input* rather than an inlined body constant:
// cmpf olt + select with the bound on input 0. This is half of the torch
// ONNX HardSigmoid decomposition; ClampOpConversion (matchOneSidedClampConstInput)
// must lower it to a single torq_hl.act clamp on the NSS slice. Without that
// the generic is left for the Host/CSS fallback, which test_linalg_ops
// forces off, so this case fails loudly.
module {
  func.func public @main(%arg0: tensor<1x1024x64xbf16>) -> tensor<1x1024x64xbf16> {
    %hi = arith.constant dense<1.000000e+00> : tensor<1xbf16>
    %e = tensor.empty() : tensor<1x1024x64xbf16>
    %r = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (0)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%hi, %arg0 : tensor<1xbf16>, tensor<1x1024x64xbf16>) outs(%e : tensor<1x1024x64xbf16>) {
    ^bb0(%b: bf16, %in: bf16, %o: bf16):
      %c = arith.cmpf olt, %in, %b : bf16
      %s = arith.select %c, %in, %b : bf16
      linalg.yield %s : bf16
    } -> tensor<1x1024x64xbf16>
    return %r : tensor<1x1024x64xbf16>
  }
}
