// Quantized 9x9 stride-1 MaxPool (DQ -> pad -> MaxPool -> Q) whose quantize clamps to [-128, 100],
// narrower than the int8 range: the clamp must be kept when the chain is folded into an int8
// maxpool.
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @main(%arg0: tensor<1x16x30x30xi8>) -> tensor<1x16x30x30xi8> {
    %cst_scale = arith.constant 0.0769420862 : f32
    %cst_zp_f32 = arith.constant -1.280000e+02 : f32
    %cst_min = arith.constant -1.280000e+02 : f32
    %cst_max = arith.constant 1.000000e+02 : f32
    %cst_neg_inf = arith.constant 0xFF800000 : f32
    %cst_i32_zp = arith.constant -128 : i32

    %0 = tensor.empty() : tensor<1x16x30x30xf32>
    %1 = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%arg0 : tensor<1x16x30x30xi8>) outs(%0 : tensor<1x16x30x30xf32>) {
    ^bb0(%in: i8, %out: f32):
      %ext = arith.extsi %in : i8 to i32
      %sub = arith.subi %ext, %cst_i32_zp : i32
      %fp = arith.sitofp %sub : i32 to f32
      %mul = arith.mulf %fp, %cst_scale : f32
      linalg.yield %mul : f32
    } -> tensor<1x16x30x30xf32>

    %padded = tensor.pad %1 low[0, 0, 4, 4] high[0, 0, 4, 4] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_neg_inf : f32
    } : tensor<1x16x30x30xf32> to tensor<1x16x38x38xf32>

    %2 = tensor.empty() : tensor<9x9xf32>
    %3 = tensor.empty() : tensor<1x16x30x30xf32>
    %4 = linalg.fill ins(%cst_neg_inf : f32) outs(%3 : tensor<1x16x30x30xf32>) -> tensor<1x16x30x30xf32>
    %5 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%padded, %2 : tensor<1x16x38x38xf32>, tensor<9x9xf32>)
      outs(%4 : tensor<1x16x30x30xf32>) -> tensor<1x16x30x30xf32>

    %6 = tensor.empty() : tensor<1x16x30x30xi8>
    %7 = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%5 : tensor<1x16x30x30xf32>) outs(%6 : tensor<1x16x30x30xi8>) {
    ^bb0(%in: f32, %out: i8):
      %div = arith.divf %in, %cst_scale : f32
      %rnd = math.roundeven %div : f32
      %add = arith.addf %rnd, %cst_zp_f32 : f32
      %clmp_lo = arith.maximumf %add, %cst_min : f32
      %clmp_hi = arith.minimumf %clmp_lo, %cst_max : f32
      %qi = arith.fptosi %clmp_hi : f32 to i8
      linalg.yield %qi : i8
    } -> tensor<1x16x30x30xi8>

    return %7 : tensor<1x16x30x30xi8>
  }
}
