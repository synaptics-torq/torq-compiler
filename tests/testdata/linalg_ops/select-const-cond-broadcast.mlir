// A constant non-splat i1 condition (causal-style mask) with
// 1-element bf16 data constants, the condition broadcast over the leading
// dim: the NPU select kernel's two-source D read overreads the staged data
// constants on hardware.
module attributes {tf_saved_model.semantics} {
  func.func @main() -> tensor<1x4x16x16xbf16> {
    %cond = arith.constant dense<
      [[[[true, false, false, false, false, false, false, false, false, false, false, false, false, false,
       false, false], [true, true, false, false, false, false, false, false, false, false, false, false,
       false, false, false, false], [true, true, true, false, false, false, false, false, false, false,
       false, false, false, false, false, false], [true, true, true, true, false, false, false, false,
       false, false, false, false, false, false, false, false], [true, true, true, true, true, false, false,
       false, false, false, false, false, false, false, false, false], [true, true, true, true, true, true,
       false, false, false, false, false, false, false, false, false, false], [true, true, true, true,
       true, true, true, false, false, false, false, false, false, false, false, false], [true, true, true,
       true, true, true, true, true, false, false, false, false, false, false, false, false], [true, true,
       true, true, true, true, true, true, true, false, false, false, false, false, false, false], [true,
       true, true, true, true, true, true, true, true, true, false, false, false, false, false, false]
      , [true, true, true, true, true, true, true, true, true, true, true, false, false, false, false,
       false], [true, true, true, true, true, true, true, true, true, true, true, true, false, false, false,
       false], [true, true, true, true, true, true, true, true, true, true, true, true, true, false, false,
       false], [true, true, true, true, true, true, true, true, true, true, true, true, true, true, false,
       false], [true, true, true, true, true, true, true, true, true, true, true, true, true, true, true,
       false], [true, true, true, true, true, true, true, true, true, true, true, true, true, true, true,
       true]]]]
    > : tensor<1x1x16x16xi1>
    %keep = arith.constant dense<0.5> : tensor<1xbf16>
    %block = arith.constant dense<-2.0> : tensor<1xbf16>
    %e = tensor.empty() : tensor<1x4x16x16xbf16>
    %r = "linalg.generic"(%cond, %keep, %block, %e) <{indexing_maps = [
      affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>,
      affine_map<(d0, d1, d2, d3) -> (0)>,
      affine_map<(d0, d1, d2, d3) -> (0)>,
      affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
    ], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 3, 1>}> ({
    ^bb0(%x: i1, %y: bf16, %z: bf16, %w: bf16):
      %s = "arith.select"(%x, %y, %z) : (i1, bf16, bf16) -> bf16
      "linalg.yield"(%s) : (bf16) -> ()
    }) : (tensor<1x1x16x16xi1>, tensor<1xbf16>, tensor<1xbf16>, tensor<1x4x16x16xbf16>) -> tensor<1x4x16x16xbf16>
    return %r : tensor<1x4x16x16xbf16>
  }
}
