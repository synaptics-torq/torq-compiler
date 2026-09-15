// Regression: same fused bf16 fully_connected, with IC=4, which is below
// wram.transposeWidth(). maxDivisor then returns IC itself, so one chunk covers the
// whole reduction and the transpose block spans it in one go.
//
// The weight is an argument, not a constant, purely to keep this file small.
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<16x4xbf16>, %arg1: tensor<4x64xbf16>) -> tensor<16x64xbf16> {
    %cst = arith.constant dense<"0x113FA13DC53CDBBC9D3CCE3E8D3EDD3D05BF833EAFBE0C3F23BF8DBD71BB2ABF5C3F3B3F6DBEC63E423EA7BF003EFBBC2A3D0ABF0ABEB7BD183F2B3E36BB443F8EBE47BE69BF493FF73EEB3EAB3E623DDD3D01BED0BDDE3C423F8E3EEFBC94BEA3BE4D3F823E0A3D31BE0EBF09BDE03E49BEE9BDE2BD603D4CBFF1BDDBBEE23E"> : tensor<64xbf16>
    %0 = tensor.empty() : tensor<16x64xf32>
    %1 = tensor.empty() : tensor<64xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%cst : tensor<64xbf16>) outs(%1 : tensor<64xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %7 = arith.extf %in : bf16 to f32
      linalg.yield %7 : f32
    } -> tensor<64xf32>
    %broadcasted = linalg.broadcast ins(%2 : tensor<64xf32>) outs(%0 : tensor<16x64xf32>) dimensions = [0]
    %3 = linalg.matmul ins(%arg0, %arg1 : tensor<16x4xbf16>, tensor<4x64xbf16>) outs(%broadcasted : tensor<16x64xf32>) -> tensor<16x64xf32>
    %4 = tensor.empty() : tensor<16x64xbf16>
    %5 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<16x64xf32>) outs(%4 : tensor<16x64xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %7 = arith.truncf %in : f32 to bf16
      linalg.yield %7 : bf16
    } -> tensor<16x64xbf16>
    return %5 : tensor<16x64xbf16>
  }
}
