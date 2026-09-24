// Regression: same fused bf16 fully_connected as fc-bias-rowblock-bf16, but with OC=48.
// OC is above the bf16 aluWidth of 32, so vectorize() pads the 16-channel remainder
// vector to a full 32, and both vectors get act_limit=4 bias groups. Keeps the OC
// remainder path covered on the fast kernel.
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<16x64xbf16>, %arg1: tensor<64x48xbf16>) -> tensor<16x48xbf16> {
    %cst = arith.constant dense<"0x7BBF0B3F273FEFBD603F04BD44BC123FCC3B69BE38BE5CBF20BD24BC1C3FA53EBFBECE3E163E483FE73E54BD063F3FBEF6BEF83D8F3E063F6EBCEE3CBA3E14BD4E3F3EBF22BF0DBE0E3F3CBD2DBFC73E10BF243FE83E413F213F0C3FB73E1FBE"> : tensor<48xbf16>
    %0 = tensor.empty() : tensor<16x48xf32>
    %1 = tensor.empty() : tensor<48xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%cst : tensor<48xbf16>) outs(%1 : tensor<48xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %7 = arith.extf %in : bf16 to f32
      linalg.yield %7 : f32
    } -> tensor<48xf32>
    %broadcasted = linalg.broadcast ins(%2 : tensor<48xf32>) outs(%0 : tensor<16x48xf32>) dimensions = [0]
    %3 = linalg.matmul ins(%arg0, %arg1 : tensor<16x64xbf16>, tensor<64x48xbf16>) outs(%broadcasted : tensor<16x48xf32>) -> tensor<16x48xf32>
    %4 = tensor.empty() : tensor<16x48xbf16>
    %5 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<16x48xf32>) outs(%4 : tensor<16x48xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %7 = arith.truncf %in : f32 to bf16
      linalg.yield %7 : bf16
    } -> tensor<16x48xbf16>
    return %5 : tensor<16x48xbf16>
  }
}
