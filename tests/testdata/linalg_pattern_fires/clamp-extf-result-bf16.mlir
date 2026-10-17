// bf16 multiply followed by a clamp whose generic converts width (clamps in
// bf16, extf the result to f32): the form the elementwise fusion produces when
// a model's final ReLU/clamp is merged with the f32 conversion of a bf16
// output. The forward float tail fold must STOP at that clamp -- absorbing it
// would end the tail on an f32 type and replace the bf16 producer's result
// with an f32-typed op (or, for producers whose result type follows the tail,
// emit a type-mismatched torq_hl op and lose the clip).
//
// Expected after torq-linalg-to-torqhl-conversion: the mul stays a
// tensor<1x4x1x4xbf16> torq_hl.mul (no tail), and the extf-ReLU generic is
// lowered on its own to a bf16-in / f32-out torq_hl.act clamp.
#id = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#sc = affine_map<(d0, d1, d2, d3) -> (0)>
module {
  func.func @main(%arg0: tensor<1x4x1x4xbf16>) -> tensor<1x4x1x4xf32> {
    %half = arith.constant dense<5.000000e-01> : tensor<1xbf16>
    %zb = arith.constant 0.000000e+00 : bf16
    %e = tensor.empty() : tensor<1x4x1x4xbf16>
    %e_0 = tensor.empty() : tensor<1x4x1x4xf32>
    %m = linalg.generic {indexing_maps = [#id, #sc, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%arg0, %half : tensor<1x4x1x4xbf16>, tensor<1xbf16>) outs(%e : tensor<1x4x1x4xbf16>) {
    ^bb0(%in: bf16, %h: bf16, %init: bf16):
      %r = arith.mulf %in, %h : bf16
      linalg.yield %r : bf16
    } -> tensor<1x4x1x4xbf16>
    %c = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%m : tensor<1x4x1x4xbf16>) outs(%e_0 : tensor<1x4x1x4xf32>) {
    ^bb0(%in: bf16, %init: f32):
      %cmp = arith.cmpf ogt, %in, %zb : bf16
      %s = arith.select %cmp, %in, %zb : bf16
      %f = arith.extf %s : bf16 to f32
      linalg.yield %f : f32
    } -> tensor<1x4x1x4xf32>
    return %c : tensor<1x4x1x4xf32>
  }
}
