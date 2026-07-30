// Quantized AveragePool QDQ fusion test.
// Shape matches Inception v3's /Mixed_5b/AveragePool: 1x192x35x35 input,
// 3x3 kernel, stride 1, symmetric pad [1,1] on H/W, output 1x192x35x35.
// Input/Output are int8; the internal DQ -> pad -> pooling_nchw_sum -> div -> Q
// chain should fold into a single int8 torq_hl.depthwise_conv2d op on NSS.
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @main(%arg0: tensor<1x192x35x35xi8>) -> tensor<1x192x35x35xi8> {
    %cst_scale = arith.constant 0.00662427722 : f32
    %cst_zp_f32 = arith.constant -2.400000e+01 : f32
    %cst_zp_i32 = arith.constant -24 : i32
    %cst_min = arith.constant -1.280000e+02 : f32
    %cst_max = arith.constant 1.270000e+02 : f32
    %cst_pad = arith.constant 0.000000e+00 : f32
    %cst_div = arith.constant 9.000000e+00 : f32

    %0 = tensor.empty() : tensor<1x192x35x35xf32>
    %1 = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%arg0 : tensor<1x192x35x35xi8>) outs(%0 : tensor<1x192x35x35xf32>) {
    ^bb0(%in: i8, %out: f32):
      %ext = arith.extsi %in : i8 to i32
      %sub = arith.subi %ext, %cst_zp_i32 : i32
      %fp = arith.sitofp %sub : i32 to f32
      %mul = arith.mulf %fp, %cst_scale : f32
      linalg.yield %mul : f32
    } -> tensor<1x192x35x35xf32>

    %padded = tensor.pad %1 low[0, 0, 1, 1] high[0, 0, 1, 1] {
    ^bb0(%arg1: index, %arg2: index, %arg3: index, %arg4: index):
      tensor.yield %cst_pad : f32
    } : tensor<1x192x35x35xf32> to tensor<1x192x37x37xf32>

    %2 = tensor.empty() : tensor<3x3xf32>
    %3 = tensor.empty() : tensor<1x192x35x35xf32>
    %4 = linalg.fill ins(%cst_pad : f32) outs(%3 : tensor<1x192x35x35xf32>) -> tensor<1x192x35x35xf32>
    %5 = linalg.pooling_nchw_sum {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>}
      ins(%padded, %2 : tensor<1x192x37x37xf32>, tensor<3x3xf32>)
      outs(%4 : tensor<1x192x35x35xf32>) -> tensor<1x192x35x35xf32>

    %6 = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%5 : tensor<1x192x35x35xf32>) outs(%3 : tensor<1x192x35x35xf32>) {
    ^bb0(%in: f32, %out: f32):
      %div = arith.divf %in, %cst_div : f32
      linalg.yield %div : f32
    } -> tensor<1x192x35x35xf32>

    %7 = tensor.empty() : tensor<1x192x35x35xi8>
    %8 = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%6 : tensor<1x192x35x35xf32>) outs(%7 : tensor<1x192x35x35xi8>) {
    ^bb0(%in: f32, %out: i8):
      %div = arith.divf %in, %cst_scale : f32
      %rnd = math.roundeven %div : f32
      %add = arith.addf %rnd, %cst_zp_f32 : f32
      %clmp_lo = arith.maximumf %add, %cst_min : f32
      %clmp_hi = arith.minimumf %clmp_lo, %cst_max : f32
      %qi = arith.fptosi %clmp_hi : f32 to i8
      linalg.yield %qi : i8
    } -> tensor<1x192x35x35xi8>

    return %8 : tensor<1x192x35x35xi8>
  }
}
