// Regression test for synaptics-torq/torq-compiler-dev#2632: a reduction large
// enough to be tiled (648x512 bf16 overflows the LRAM tiling budget) used to
// route its per-chunk partial-sum generic to the f32 host fallback, whose strided
// partial-buffer column the runtime copied flat before b35fdc35 (rows of the outer
// dim doubled/zeroed). The hardware-supported reduce sum must stay on the NPU.
module {
  func.func @main(%a: tensor<648x512xbf16>) -> tensor<648xf32> {
    %c = arith.constant 0.0 : f32
    %e = tensor.empty() : tensor<648xf32>
    %f = linalg.fill ins(%c : f32) outs(%e : tensor<648xf32>) -> tensor<648xf32>
    %r = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0)>], iterator_types = ["parallel", "reduction"]} ins(%a : tensor<648x512xbf16>) outs(%f : tensor<648xf32>) {
    ^bb0(%in: bf16, %init: f32):
      %x = arith.extf %in : bf16 to f32
      %s = arith.addf %x, %init : f32
      linalg.yield %s : f32
    } -> tensor<648xf32>
    return %r : tensor<648xf32>
  }
}
