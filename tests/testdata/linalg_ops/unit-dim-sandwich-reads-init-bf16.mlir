// Regression: a collapse -> elementwise -> expand unit-dim sandwich whose
// elementwise body reads its init must keep the init; it cannot be re-emitted
// at the outer shape on a fresh init.
#id = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<1x1x256xbf16>, %arg1: tensor<1x256xbf16>) -> tensor<1x1x256xbf16> {
    %0 = tensor.collapse_shape %arg0 [[0, 1], [2]] : tensor<1x1x256xbf16> into tensor<1x256xbf16>
    %1 = linalg.generic {indexing_maps = [#id, #id], iterator_types = ["parallel", "parallel"]}
        ins(%0 : tensor<1x256xbf16>) outs(%arg1 : tensor<1x256xbf16>) {
    ^bb0(%a: bf16, %out: bf16):
      %s = arith.addf %a, %out : bf16
      linalg.yield %s : bf16
    } -> tensor<1x256xbf16>
    %2 = tensor.expand_shape %1 [[0, 1], [2]] output_shape [1, 1, 256] : tensor<1x256xbf16> into tensor<1x1x256xbf16>
    return %2 : tensor<1x1x256xbf16>
  }
}
