// Regression: an NCHW->NHWC layout transpose feeding a rank-reducing extract_slice.
// The slice and the add after it inherit the transpose's 4D propagation state but
// have rank-2 results. OptimizeTransposeLayout cannot adapt a 4D permutation to rank 2,
// and it used to assert in Permutation::reverse(); those ops must be left in place.
module {
  func.func @main(%input: tensor<1x8x4x4xbf16>, %bias: tensor<4x8xbf16>) -> tensor<4x8xbf16> {
    %e = tensor.empty() : tensor<1x4x4x8xbf16>
    %t = linalg.transpose ins(%input : tensor<1x8x4x4xbf16>) outs(%e : tensor<1x4x4x8xbf16>) permutation = [0, 2, 3, 1]
    %s = tensor.extract_slice %t[0, 1, 0, 0] [1, 1, 4, 8] [1, 1, 1, 1] : tensor<1x4x4x8xbf16> to tensor<4x8xbf16>
    %e2 = tensor.empty() : tensor<4x8xbf16>
    %r = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
    } ins(%s, %bias : tensor<4x8xbf16>, tensor<4x8xbf16>) outs(%e2 : tensor<4x8xbf16>) {
    ^bb0(%a: bf16, %b: bf16, %out: bf16):
      %m = arith.addf %a, %b : bf16
      linalg.yield %m : bf16
    } -> tensor<4x8xbf16>
    return %r : tensor<4x8xbf16>
  }
}
