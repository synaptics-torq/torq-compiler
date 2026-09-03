// Weight-first quantized batch_matmul with a ternary (i2-range) i8 weight constant.
// torq-narrow-weight-const packs the i8 weight to i2 and the matmul weight peel streams
// it on the W-bus (weight_format = SI); the result must match the bf16 reference.
// Mirrors the whisper decoder FFN form: Cast(Wq[N,K]) unsqueezed to a same-rank batched
// MatMul with the weight as operand 0, reached through a tensor.expand_shape.
// Tolerance covers bf16 accumulation-order rounding between the torq systolic path
// and the llvm-cpu reference (the narrowing itself is value-preserving).
// TORQ_FP_MAX_TOL: 0.1
// TORQ_FP_AVG_TOL: 0.02
#id2 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%x: tensor<1x16x4xbf16>) -> tensor<1x8x4xbf16> {
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
