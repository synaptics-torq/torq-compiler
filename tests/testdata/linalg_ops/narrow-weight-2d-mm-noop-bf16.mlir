// A plain 2D linalg.matmul with an i2-range i8 weight constant. The weight fits i2, but a
// 2D matmul lowers to fully_connected, which does not peel the weight onto the W-bus, so
// torq-narrow-weight-const must leave it as i8 (narrowing it would leave an unplaceable
// sub-byte unpack). Regression guard: this must compile and match the bf16 reference.
// TORQ_FP_MAX_TOL: 0.1
// TORQ_FP_AVG_TOL: 0.02
#id2 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%x: tensor<16x4xbf16>) -> tensor<8x4xbf16> {
    %w = arith.constant dense<[
      [-1, 0, 1, 1, -1, 0, 1, -1, 0, 1, -1, 1, 0, -1, 1, 0],
      [ 0, 1, -1, 1, 0, -1, 1, 0, 1, -1, 0, 0, 1, 1, -1, 0],
      [ 1, -1, 0, 0, 1, 1, -1, 0, -1, 0, 1, -1, 1, 0, 0, 1],
      [-1, 0, 1, -1, 1, 0, 0, 1, 0, 1, -1, 1, 0, -1, 1, 0],
      [ 1, 0, -1, 1, -1, 0, 1, -1, 0, 1, -1, 1, 0, -1, 1, 0],
      [ 0, 1, -1, 1, 0, -1, 1, 0, 1, -1, 0, 0, 1, 1, -1, 0],
      [ 1, -1, 0, 0, 1, 1, -1, 0, -1, 0, 1, -1, 1, 0, 0, 1],
      [-1, 0, 1, -1, 1, 0, 0, 1, 0, 1, -1, 1, 0, -1, 1, 0]
    ]> : tensor<8x16xi8>
    %wi = tensor.empty() : tensor<8x16xbf16>
    %wbf = linalg.generic {indexing_maps = [#id2, #id2], iterator_types = ["parallel", "parallel"]}
        ins(%w : tensor<8x16xi8>) outs(%wi : tensor<8x16xbf16>) {
    ^bb0(%a: i8, %b: bf16):
      %f = arith.sitofp %a : i8 to bf16
      linalg.yield %f : bf16
    } -> tensor<8x16xbf16>
    %zero = arith.constant 0.000000e+00 : bf16
    %init0 = tensor.empty() : tensor<8x4xbf16>
    %init = linalg.fill ins(%zero : bf16) outs(%init0 : tensor<8x4xbf16>) -> tensor<8x4xbf16>
    %r = linalg.matmul ins(%wbf, %x : tensor<8x16xbf16>, tensor<16x4xbf16>)
        outs(%init : tensor<8x4xbf16>) -> tensor<8x4xbf16>
    return %r : tensor<8x4xbf16>
  }
}
