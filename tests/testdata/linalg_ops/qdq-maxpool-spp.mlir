// Quantized YOLO SPP block: one dequantize feeding 5x5, 9x9 and 13x13 stride-1 max pools, each
// followed by a quantize with the same scale and zero point. The quantization folds away (int8
// maxpools) and the 9x9 and 13x13 pools are computed from the 5x5 one: 3 chained int8 5x5 maxpools.
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @main(%arg0: tensor<1x64x20x20xi8>) -> (tensor<1x64x20x20xi8>, tensor<1x64x20x20xi8>, tensor<1x64x20x20xi8>) {
    %cst_scale = arith.constant 0.0769420862 : f32
    %cst_zp_f32 = arith.constant -1.280000e+02 : f32
    %cst_min = arith.constant -1.280000e+02 : f32
    %cst_max = arith.constant 1.270000e+02 : f32
    %cst_neg_inf = arith.constant 0xFF800000 : f32
    %cst_i32_zp = arith.constant -128 : i32
    %e = tensor.empty() : tensor<1x64x20x20xf32>
    %dq = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%arg0 : tensor<1x64x20x20xi8>) outs(%e : tensor<1x64x20x20xf32>) {
    ^bb0(%in: i8, %out: f32):
      %ext = arith.extsi %in : i8 to i32
      %sub = arith.subi %ext, %cst_i32_zp : i32
      %fp = arith.sitofp %sub : i32 to f32
      %mul = arith.mulf %fp, %cst_scale : f32
      linalg.yield %mul : f32
    } -> tensor<1x64x20x20xf32>

    %pad0 = tensor.pad %dq low[0, 0, 2, 2] high[0, 0, 2, 2] {
    ^bb0(%a0: index, %b0: index, %c0: index, %d0: index):
      tensor.yield %cst_neg_inf : f32
    } : tensor<1x64x20x20xf32> to tensor<1x64x24x24xf32>
    %k0 = tensor.empty() : tensor<5x5xf32>
    %e0 = tensor.empty() : tensor<1x64x20x20xf32>
    %f0 = linalg.fill ins(%cst_neg_inf : f32) outs(%e0 : tensor<1x64x20x20xf32>) -> tensor<1x64x20x20xf32>
    %pool0 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%pad0, %k0 : tensor<1x64x24x24xf32>, tensor<5x5xf32>)
      outs(%f0 : tensor<1x64x20x20xf32>) -> tensor<1x64x20x20xf32>
    %qe0 = tensor.empty() : tensor<1x64x20x20xi8>
    %q0 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%pool0 : tensor<1x64x20x20xf32>) outs(%qe0 : tensor<1x64x20x20xi8>) {
    ^bb0(%in: f32, %out: i8):
      %div = arith.divf %in, %cst_scale : f32
      %rnd = math.roundeven %div : f32
      %add = arith.addf %rnd, %cst_zp_f32 : f32
      %lo = arith.maximumf %add, %cst_min : f32
      %hi = arith.minimumf %lo, %cst_max : f32
      %qi = arith.fptosi %hi : f32 to i8
      linalg.yield %qi : i8
    } -> tensor<1x64x20x20xi8>
    %pad1 = tensor.pad %dq low[0, 0, 4, 4] high[0, 0, 4, 4] {
    ^bb0(%a1: index, %b1: index, %c1: index, %d1: index):
      tensor.yield %cst_neg_inf : f32
    } : tensor<1x64x20x20xf32> to tensor<1x64x28x28xf32>
    %k1 = tensor.empty() : tensor<9x9xf32>
    %e1 = tensor.empty() : tensor<1x64x20x20xf32>
    %f1 = linalg.fill ins(%cst_neg_inf : f32) outs(%e1 : tensor<1x64x20x20xf32>) -> tensor<1x64x20x20xf32>
    %pool1 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%pad1, %k1 : tensor<1x64x28x28xf32>, tensor<9x9xf32>)
      outs(%f1 : tensor<1x64x20x20xf32>) -> tensor<1x64x20x20xf32>
    %qe1 = tensor.empty() : tensor<1x64x20x20xi8>
    %q1 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%pool1 : tensor<1x64x20x20xf32>) outs(%qe1 : tensor<1x64x20x20xi8>) {
    ^bb0(%in: f32, %out: i8):
      %div = arith.divf %in, %cst_scale : f32
      %rnd = math.roundeven %div : f32
      %add = arith.addf %rnd, %cst_zp_f32 : f32
      %lo = arith.maximumf %add, %cst_min : f32
      %hi = arith.minimumf %lo, %cst_max : f32
      %qi = arith.fptosi %hi : f32 to i8
      linalg.yield %qi : i8
    } -> tensor<1x64x20x20xi8>
    %pad2 = tensor.pad %dq low[0, 0, 6, 6] high[0, 0, 6, 6] {
    ^bb0(%a2: index, %b2: index, %c2: index, %d2: index):
      tensor.yield %cst_neg_inf : f32
    } : tensor<1x64x20x20xf32> to tensor<1x64x32x32xf32>
    %k2 = tensor.empty() : tensor<13x13xf32>
    %e2 = tensor.empty() : tensor<1x64x20x20xf32>
    %f2 = linalg.fill ins(%cst_neg_inf : f32) outs(%e2 : tensor<1x64x20x20xf32>) -> tensor<1x64x20x20xf32>
    %pool2 = linalg.pooling_nchw_max {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%pad2, %k2 : tensor<1x64x32x32xf32>, tensor<13x13xf32>)
      outs(%f2 : tensor<1x64x20x20xf32>) -> tensor<1x64x20x20xf32>
    %qe2 = tensor.empty() : tensor<1x64x20x20xi8>
    %q2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel", "parallel", "parallel"]}
      ins(%pool2 : tensor<1x64x20x20xf32>) outs(%qe2 : tensor<1x64x20x20xi8>) {
    ^bb0(%in: f32, %out: i8):
      %div = arith.divf %in, %cst_scale : f32
      %rnd = math.roundeven %div : f32
      %add = arith.addf %rnd, %cst_zp_f32 : f32
      %lo = arith.maximumf %add, %cst_min : f32
      %hi = arith.minimumf %lo, %cst_max : f32
      %qi = arith.fptosi %hi : f32 to i8
      linalg.yield %qi : i8
    } -> tensor<1x64x20x20xi8>
    return %q0, %q1, %q2 : tensor<1x64x20x20xi8>, tensor<1x64x20x20xi8>, tensor<1x64x20x20xi8>
  }
}
