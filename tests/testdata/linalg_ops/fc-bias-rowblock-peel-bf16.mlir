// Regression: same fused bf16 fully_connected as fc-bias-rowblock-bf16, but with
// N=13 rows. 12 rows go to the row-blocked (fc-fast) kernel and the remaining row is
// peeled onto the scalar kernel, so this covers the peel offset.
//
// The weight is an argument, not a constant, purely to keep this file small.
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<13x64xbf16>, %arg1: tensor<64x64xbf16>) -> tensor<13x64xbf16> {
    %cst = arith.constant dense<"0x7BBF0B3F273FEFBD603F04BD44BC123FCC3B69BE38BE5CBF20BD24BC1C3FA53EBFBECE3E163E483FE73E54BD063F3FBEF6BEF83D8F3E063F6EBCEE3CBA3E14BD4E3F3EBF22BF0DBE0E3F3CBD2DBFC73E10BF243FE83E413F213F0C3FB73E1FBE4FBDE73BCB3D8ABEBBBE363F16BF0A3F823F02BF323F69BD3C3F90BE263FB0BE"> : tensor<64xbf16>
    %0 = tensor.empty() : tensor<13x64xf32>
    %1 = tensor.empty() : tensor<64xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%cst : tensor<64xbf16>) outs(%1 : tensor<64xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %7 = arith.extf %in : bf16 to f32
      linalg.yield %7 : f32
    } -> tensor<64xf32>
    %broadcasted = linalg.broadcast ins(%2 : tensor<64xf32>) outs(%0 : tensor<13x64xf32>) dimensions = [0]
    %3 = linalg.matmul ins(%arg0, %arg1 : tensor<13x64xbf16>, tensor<64x64xbf16>) outs(%broadcasted : tensor<13x64xf32>) -> tensor<13x64xf32>
    %4 = tensor.empty() : tensor<13x64xbf16>
    %5 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<13x64xf32>) outs(%4 : tensor<13x64xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %7 = arith.truncf %in : f32 to bf16
      linalg.yield %7 : bf16
    } -> tensor<13x64xbf16>
    return %5 : tensor<13x64xbf16>
  }
}
