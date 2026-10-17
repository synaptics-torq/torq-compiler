// Per-channel mul + per-channel bias + scalar add + ReLU clamp, all bf16.
// The mulf and the per-channel addf share a fuse group (the attributes the
// MarkPatternsForTileAndFuse / CoalesceElementwiseChains marking passes emit)
// so MulPerChannelBiasPattern can fold the whole chain into one torq_hl.mul:
// the per-channel bias constant is rebuilt in f32 with the tail's scalar add
// absorbed, and the ReLU becomes the ACT clip range.
//
// Expected after torq-linalg-to-torqhl-conversion: exactly one torq_hl.mul
// (bias dense<[0.6, 0.7, 0.8, 0.9]> : tensor<4xf32>, output_min = 0, no
// linalg.generic left). If the tail's scalar add is lost or the add/clamp
// generics are left dangling, the conversion misbehaves or crashes.
#id = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#ch = affine_map<(d0, d1, d2, d3) -> (d1)>
#sc = affine_map<(d0, d1, d2, d3) -> (0)>
module {
  func.func @main(%arg0: tensor<1x4x1x4xbf16>) -> tensor<1x4x1x4xbf16> {
    %scale = arith.constant dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xbf16>
    %bias = arith.constant dense<[0.100000e+00, 0.200000e+00, 0.300000e+00, 0.400000e+00]> : tensor<4xbf16>
    %k = arith.constant dense<0.500000e+00> : tensor<1xbf16>
    %zero = arith.constant dense<0.000000e+00> : tensor<1xbf16>
    %e = tensor.empty() : tensor<1x4x1x4xbf16>
    %e_0 = tensor.empty() : tensor<1x4x1x4xbf16>
    %e_1 = tensor.empty() : tensor<1x4x1x4xbf16>
    %e_2 = tensor.empty() : tensor<1x4x1x4xbf16>
    %m = linalg.generic {indexing_maps = [#id, #ch, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"], "torq-fuse-group-id" = 7 : i64, "torq-fuse-group" = [7 : i64]} ins(%arg0, %scale : tensor<1x4x1x4xbf16>, tensor<4xbf16>) outs(%e : tensor<1x4x1x4xbf16>) {
    ^bb0(%in: bf16, %w: bf16, %init: bf16):
      %r = arith.mulf %in, %w : bf16
      linalg.yield %r : bf16
    } -> tensor<1x4x1x4xbf16>
    %b = linalg.generic {indexing_maps = [#id, #ch, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"], "torq-fuse-group" = [7 : i64]} ins(%m, %bias : tensor<1x4x1x4xbf16>, tensor<4xbf16>) outs(%e_0 : tensor<1x4x1x4xbf16>) {
    ^bb0(%in: bf16, %bi: bf16, %init: bf16):
      %r = arith.addf %in, %bi : bf16
      linalg.yield %r : bf16
    } -> tensor<1x4x1x4xbf16>
    %s = linalg.generic {indexing_maps = [#id, #sc, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%b, %k : tensor<1x4x1x4xbf16>, tensor<1xbf16>) outs(%e_1 : tensor<1x4x1x4xbf16>) {
    ^bb0(%in: bf16, %kk: bf16, %init: bf16):
      %r = arith.addf %in, %kk : bf16
      linalg.yield %r : bf16
    } -> tensor<1x4x1x4xbf16>
    %c = linalg.generic {indexing_maps = [#sc, #id, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%zero, %s : tensor<1xbf16>, tensor<1x4x1x4xbf16>) outs(%e_2 : tensor<1x4x1x4xbf16>) {
    ^bb0(%z: bf16, %in: bf16, %init: bf16):
      %cmp = arith.cmpf ogt, %in, %z : bf16
      %r = arith.select %cmp, %in, %z : bf16
      linalg.yield %r : bf16
    } -> tensor<1x4x1x4xbf16>
    return %c : tensor<1x4x1x4xbf16>
  }
}
