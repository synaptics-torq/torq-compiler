// Regression: bf16 matmul + per-output-channel bias, which Conv2DMatmulPattern fuses
// into one torq_hl.fully_connected. Unlike fc-bias-rowblock-tiled-bf16.mlir, where
// TileAndFuse splits the columns and every tile keeps all 128 rows, N is small here, so
// the search splits the OUTPUT ROWS: 128 rows land on a tile of 19, which is 7 tiles.
// 19 is not a multiple of the kernel row block of 4, so each tile is lowered twice: the
// row-blocked kernel takes 16 rows and a second block takes the other 3. This is the only
// case here that runs the kernel on a row range that is not the whole tensor.
//
// The weight is an argument, not a constant, purely to keep this file small.
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<128x3072xbf16>, %arg1: tensor<3072x64xbf16>) -> tensor<128x64xbf16> {
    %cst = arith.constant dense<"0x003E0ABE143E1EBE283E33BE3D3E47BE513E5CBE663E70BE7A3E82BE873E8CBE913E00BE0A3E14BE1E3E28BE333E3DBE473E51BE5C3E66BE703E7ABE823E87BE8C3E91BE003E0ABE143E1EBE283E33BE3D3E47BE513E5CBE663E70BE7A3E82BE873E8CBE913E00BE0A3E14BE1E3E28BE333E3DBE473E51BE5C3E66BE703E7ABE"> : tensor<64xbf16>
    %0 = tensor.empty() : tensor<128x64xf32>
    %1 = tensor.empty() : tensor<64xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%cst : tensor<64xbf16>) outs(%1 : tensor<64xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %7 = arith.extf %in : bf16 to f32
      linalg.yield %7 : f32
    } -> tensor<64xf32>
    %broadcasted = linalg.broadcast ins(%2 : tensor<64xf32>) outs(%0 : tensor<128x64xf32>) dimensions = [0]
    %3 = linalg.matmul ins(%arg0, %arg1 : tensor<128x3072xbf16>, tensor<3072x64xbf16>) outs(%broadcasted : tensor<128x64xf32>) -> tensor<128x64xf32>
    %4 = tensor.empty() : tensor<128x64xbf16>
    %5 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<128x64xf32>) outs(%4 : tensor<128x64xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %7 = arith.truncf %in : f32 to bf16
      linalg.yield %7 : bf16
    } -> tensor<128x64xbf16>
    return %5 : tensor<128x64xbf16>
  }
}
