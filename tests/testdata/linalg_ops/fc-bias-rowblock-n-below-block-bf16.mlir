// Regression: same fused bf16 fully_connected as fc-bias-rowblock-bf16, with N=3 rows.
// That is fewer than wram.transposeHeight(), so the kernel builds one short block of 3 rows
// and still runs in a single pass, the way lowerToFastMatmul() handles M < 4. Guards the
// small-row end of the row blocking.
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<3x64xbf16>, %arg1: tensor<64x64xbf16>) -> tensor<3x64xbf16> {
    %cst = arith.constant dense<"0x003E0ABE143E1EBE283E33BE3D3E47BE513E5CBE663E70BE7A3E82BE873E8CBE913E00BE0A3E14BE1E3E28BE333E3DBE473E51BE5C3E66BE703E7ABE823E87BE8C3E91BE003E0ABE143E1EBE283E33BE3D3E47BE513E5CBE663E70BE7A3E82BE873E8CBE913E00BE0A3E14BE1E3E28BE333E3DBE473E51BE5C3E66BE703E7ABE"> : tensor<64xbf16>
    %0 = tensor.empty() : tensor<3x64xf32>
    %1 = tensor.empty() : tensor<64xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%cst : tensor<64xbf16>) outs(%1 : tensor<64xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %7 = arith.extf %in : bf16 to f32
      linalg.yield %7 : f32
    } -> tensor<64xf32>
    %broadcasted = linalg.broadcast ins(%2 : tensor<64xf32>) outs(%0 : tensor<3x64xf32>) dimensions = [0]
    %3 = linalg.matmul ins(%arg0, %arg1 : tensor<3x64xbf16>, tensor<64x64xbf16>) outs(%broadcasted : tensor<3x64xf32>) -> tensor<3x64xf32>
    %4 = tensor.empty() : tensor<3x64xbf16>
    %5 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<3x64xf32>) outs(%4 : tensor<3x64xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %7 = arith.truncf %in : f32 to bf16
      linalg.yield %7 : bf16
    } -> tensor<3x64xbf16>
    return %5 : tensor<3x64xbf16>
  }
}
