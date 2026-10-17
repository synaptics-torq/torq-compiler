// Conv1D (5D generic form) followed by the float ACT tail: a f32->bf16 width
// conversion, a scalar const add, and a ReLU clamp, all over the [N, F, 1, Ow,
// Kw] 5D result, then the Kw reduce. LinalgGenericConv1DToTorqHLConv1DPattern
// (pre-conversion) must absorb the whole tail into one torq_hl.conv1d: the
// scalar add becomes the f32 per-channel bias (F=2), the ReLU the clip range,
// and the conv output is bf16.
//
// The constant-evaluation the per-channel bias fold performs needs the full
// compile context, so this case is checked via torq-compile
// (--mlir-print-ir-after=torq-linalg-to-torqhl-pre-conversion,
// --torq-disable-slicing keeps the chain in one dispatch), not iree-opt.
#id = affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
#in = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d5, d2, d3 + d4)>
#wt = affine_map<(d0, d1, d2, d3, d4, d5) -> (d1, d5, d2, d4)>
#out = affine_map<(d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4)>
#sc = affine_map<(d0, d1, d2, d3, d4) -> (0)>
#sc0 = affine_map<(d0, d1, d2, d3, d4) -> ()>
module {
  func.func @main(%arg0: tensor<1x4x1x5xbf16>, %arg1: tensor<2x4x1x3xbf16>) -> tensor<1x2x3xbf16> {
    %k = arith.constant dense<0.500000e+00> : tensor<1xbf16>
    %zb = arith.constant dense<0.000000e+00> : tensor<bf16>
    %c0 = arith.constant 0.000000e+00 : f32
    %e = tensor.empty() : tensor<1x2x1x3x3xf32>
    %acc0 = linalg.fill ins(%c0 : f32) outs(%e : tensor<1x2x1x3x3xf32>) -> tensor<1x2x1x3x3xf32>
    %0 = linalg.generic {indexing_maps = [#in, #wt, #out], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction"]} ins(%arg0, %arg1 : tensor<1x4x1x5xbf16>, tensor<2x4x1x3xbf16>) outs(%acc0 : tensor<1x2x1x3x3xf32>) {
    ^bb0(%in: bf16, %w: bf16, %acc: f32):
      %i = arith.extf %in : bf16 to f32
      %w0 = arith.extf %w : bf16 to f32
      %m = arith.mulf %i, %w0 : f32
      %s = arith.addf %m, %acc : f32
      linalg.yield %s : f32
    } -> tensor<1x2x1x3x3xf32>
    %e1 = tensor.empty() : tensor<1x2x1x3x3xbf16>
    %1 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%0 : tensor<1x2x1x3x3xf32>) outs(%e1 : tensor<1x2x1x3x3xbf16>) {
    ^bb0(%in: f32, %init: bf16):
      %r = arith.truncf %in : f32 to bf16
      linalg.yield %r : bf16
    } -> tensor<1x2x1x3x3xbf16>
    %e2 = tensor.empty() : tensor<1x2x1x3x3xbf16>
    %2 = linalg.generic {indexing_maps = [#id, #sc, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%1, %k : tensor<1x2x1x3x3xbf16>, tensor<1xbf16>) outs(%e2 : tensor<1x2x1x3x3xbf16>) {
    ^bb0(%in: bf16, %kk: bf16, %init: bf16):
      %r = arith.addf %in, %kk : bf16
      linalg.yield %r : bf16
    } -> tensor<1x2x1x3x3xbf16>
    %e3 = tensor.empty() : tensor<1x2x1x3x3xbf16>
    %3 = linalg.generic {indexing_maps = [#sc0, #id, #id], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel"]} ins(%zb, %2 : tensor<bf16>, tensor<1x2x1x3x3xbf16>) outs(%e3 : tensor<1x2x1x3x3xbf16>) {
    ^bb0(%z: bf16, %in: bf16, %init: bf16):
      %cmp = arith.cmpf ogt, %in, %z : bf16
      %r = arith.select %cmp, %in, %z : bf16
      linalg.yield %r : bf16
    } -> tensor<1x2x1x3x3xbf16>
    %e4 = tensor.empty() : tensor<1x2x1x3xbf16>
    %4 = linalg.reduce ins(%3 : tensor<1x2x1x3x3xbf16>) outs(%e4 : tensor<1x2x1x3xbf16>) dimensions = [4]
      (%a: bf16, %b: bf16) {
        %r = arith.addf %a, %b : bf16
        linalg.yield %r : bf16
      }
    %5 = tensor.collapse_shape %4 [[0], [1], [2, 3]] : tensor<1x2x1x3xbf16> into tensor<1x2x3xbf16>
    return %5 : tensor<1x2x3xbf16>
  }
}
