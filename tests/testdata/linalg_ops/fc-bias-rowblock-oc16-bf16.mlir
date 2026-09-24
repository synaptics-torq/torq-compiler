// Regression: same fused bf16 fully_connected as fc-bias-rowblock-bf16, but with OC=16.
// That is the tGRU hidden size, and the OC vector is narrower than aluWidth. Sizing the
// bias ACT-group dim from the vector would give 2 groups, while outer-product PRAM
// densifies that dim to act_limit=4, so DEBR indexed past the record block. The fast
// kernel sizes the group dim from aluWidth instead and keeps this shape on fc-fast.
//
// The weight is an argument, not a constant, purely to keep this file small.
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<16x64xbf16>, %arg1: tensor<64x16xbf16>) -> tensor<16x16xbf16> {
    %cst = arith.constant dense<"0x7BBF0B3F273FEFBD603F04BD44BC123FCC3B69BE38BE5CBF20BD24BC1C3FA53E"> : tensor<16xbf16>
    %0 = tensor.empty() : tensor<16x16xf32>
    %1 = tensor.empty() : tensor<16xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%cst : tensor<16xbf16>) outs(%1 : tensor<16xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %7 = arith.extf %in : bf16 to f32
      linalg.yield %7 : f32
    } -> tensor<16xf32>
    %broadcasted = linalg.broadcast ins(%2 : tensor<16xf32>) outs(%0 : tensor<16x16xf32>) dimensions = [0]
    %3 = linalg.matmul ins(%arg0, %arg1 : tensor<16x64xbf16>, tensor<64x16xbf16>) outs(%broadcasted : tensor<16x16xf32>) -> tensor<16x16xf32>
    %4 = tensor.empty() : tensor<16x16xbf16>
    %5 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<16x16xf32>) outs(%4 : tensor<16x16xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %7 = arith.truncf %in : f32 to bf16
      linalg.yield %7 : bf16
    } -> tensor<16x16xbf16>
    return %5 : tensor<16x16xbf16>
  }
}
