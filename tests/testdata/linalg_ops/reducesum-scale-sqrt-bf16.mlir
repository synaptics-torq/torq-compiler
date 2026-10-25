// Variance-like chain: sum of squares over the channel dim, a scalar multiply, then sqrt.
// sqrt lowers to x * rsqrt(x), whose Newton step has 1.5 - t. Inside the fused dispatch t is
// a non-dense tile, and the non-dense scalar add kernel used to drop the -1 weight of c - x.
//
// Compiled NSS-only (--torq-disable-host --torq-disable-css).
#id = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @main(%arg0: tensor<1x12x72x196xbf16>) -> tensor<1x1x72x196xbf16> {
    %zero = arith.constant 0.000000e+00 : bf16
    %inv = arith.constant 8.349609e-02 : bf16
    %eps = arith.constant 9.999990e-06 : bf16
    %e0 = tensor.empty() : tensor<1x12x72x196xbf16>
    %sq = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0 : tensor<1x12x72x196xbf16>) outs(%e0 : tensor<1x12x72x196xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %m = arith.mulf %in, %in : bf16
      linalg.yield %m : bf16
    } -> tensor<1x12x72x196xbf16>
    %e1 = tensor.empty() : tensor<1x1x72x196xbf16>
    %f = linalg.fill ins(%zero : bf16) outs(%e1 : tensor<1x1x72x196xbf16>) -> tensor<1x1x72x196xbf16>
    %sum = linalg.generic {indexing_maps = [#id, affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>], iterator_types = ["parallel", "reduction", "parallel", "parallel"]} ins(%sq : tensor<1x12x72x196xbf16>) outs(%f : tensor<1x1x72x196xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %a = arith.addf %in, %out : bf16
      linalg.yield %a : bf16
    } -> tensor<1x1x72x196xbf16>
    %e2 = tensor.empty() : tensor<1x1x72x196xbf16>
    %mean = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%sum : tensor<1x1x72x196xbf16>) outs(%e2 : tensor<1x1x72x196xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %m = arith.mulf %in, %inv : bf16
      linalg.yield %m : bf16
    } -> tensor<1x1x72x196xbf16>
    %e3 = tensor.empty() : tensor<1x1x72x196xbf16>
    %ve = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%mean : tensor<1x1x72x196xbf16>) outs(%e3 : tensor<1x1x72x196xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %a = arith.addf %in, %eps : bf16
      linalg.yield %a : bf16
    } -> tensor<1x1x72x196xbf16>
    %e4 = tensor.empty() : tensor<1x1x72x196xbf16>
    %r = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%ve : tensor<1x1x72x196xbf16>) outs(%e4 : tensor<1x1x72x196xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %s = math.sqrt %in : bf16
      linalg.yield %s : bf16
    } -> tensor<1x1x72x196xbf16>
    return %r : tensor<1x1x72x196xbf16>
  }
}
