// Weight-first quantized batch_matmul with an i4-range i8 weight constant (values in
// [-8,7], not all in [-2,1], so torq-narrow-weight-const packs to i4 rather than i2).
// The peel streams the packed weight on the W-bus (weight_format = SI); the result must
// match the bf16 reference.
// TORQ_FP_MAX_TOL: 0.1
// TORQ_FP_AVG_TOL: 0.02
#id2 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%x: tensor<1x16x4xbf16>) -> tensor<1x8x4xbf16> {
    %w = arith.constant dense<[
      [-8, 7, 5, -6, 3, -1, 2, 0, -4, 6, -7, 1, 4, -3, 7, -5],
      [ 6, -5, 3, 7, -8, 2, -1, 4, 0, -6, 5, -3, 1, 7, -4, 2],
      [ 1, -7, 4, 0, 5, 6, -8, 2, -3, 7, 1, -5, 3, -2, 0, 6],
      [-6, 2, -3, 5, 7, -1, 0, 4, -8, 1, 6, -7, 3, 2, -4, 5],
      [ 7, -4, 1, -8, 6, 0, 3, -5, 2, 4, -6, 5, -1, 7, -3, 0],
      [ 0, 5, -7, 2, 1, -6, 4, 7, -2, -8, 3, 6, -5, 1, 0, -4],
      [ 4, -1, 6, -3, 0, 7, -8, 1, 5, -6, 2, 3, -7, 0, 6, -2],
      [-5, 3, 0, 7, -4, 1, 6, -8, 2, 5, -3, 0, 4, -6, 7, 1]
    ]> : tensor<8x16xi8>
    %wi = tensor.empty() : tensor<8x16xbf16>
    %wbf = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
        ins(%w : tensor<8x16xi8>) outs(%wi : tensor<8x16xbf16>) {
    ^bb0(%a: i8, %b: bf16):
      %f = arith.sitofp %a : i8 to bf16
      linalg.yield %f : bf16
    } -> tensor<8x16xbf16>
    %we = tensor.expand_shape %wbf [[0, 1], [2]] output_shape [1, 8, 16]
        : tensor<8x16xbf16> into tensor<1x8x16xbf16>
    %zero = arith.constant 0.000000e+00 : bf16
    %init0 = tensor.empty() : tensor<1x8x4xbf16>
    %init = linalg.fill ins(%zero : bf16) outs(%init0 : tensor<1x8x4xbf16>) -> tensor<1x8x4xbf16>
    %r = linalg.batch_matmul ins(%we, %x : tensor<1x8x16xbf16>, tensor<1x16x4xbf16>)
        outs(%init : tensor<1x8x4xbf16>) -> tensor<1x8x4xbf16>
    return %r : tensor<1x8x4xbf16>
  }
}
