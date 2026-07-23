// Quantized ReduceMean QDQ fusion test.
// Shape: 1x4x7x7 input, reduce over axes [2,3] with keepdims, output 1x4x1x1.
// Input/Output are int8; the internal DQ -> sum -> div -> Q chain should fold
// into a single int8 torq_hl.reduce_mean op on NSS.
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map_reduce = affine_map<(d0, d1, d2, d3) -> (d0, d1)>
#map2d = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<1x4x7x7xi8>) -> tensor<1x4x1x1xi8> {
    %cst_scale = arith.constant 1.000000e-01 : f32
    %cst_zp_f32 = arith.constant -1.280000e+02 : f32
    %cst_min = arith.constant -1.280000e+02 : f32
    %cst_max = arith.constant 1.270000e+02 : f32
    %cst_i32_zp = arith.constant -128 : i32
    %cst_count = arith.constant 4.900000e+01 : f32
    %cst_out_scale = arith.constant 0.00999999977 : f32
    %cst_zero = arith.constant 0.000000e+00 : f32

    %0 = tensor.empty() : tensor<1x4x7x7xf32>
    %1 = linalg.generic {
        indexing_maps = [#map, #map],
        iterator_types = ["parallel", "parallel", "parallel", "parallel"]
      } ins(%arg0 : tensor<1x4x7x7xi8>) outs(%0 : tensor<1x4x7x7xf32>) {
    ^bb0(%in: i8, %out: f32):
      %ext = arith.extsi %in : i8 to i32
      %sub = arith.subi %ext, %cst_i32_zp : i32
      %fp = arith.sitofp %sub : i32 to f32
      %mul = arith.mulf %fp, %cst_scale : f32
      linalg.yield %mul : f32
    } -> tensor<1x4x7x7xf32>

    %2 = tensor.empty() : tensor<1x4xf32>
    %3 = linalg.fill ins(%cst_zero : f32) outs(%2 : tensor<1x4xf32>) -> tensor<1x4xf32>
    %4 = linalg.generic {
        indexing_maps = [#map, #map_reduce],
        iterator_types = ["parallel", "parallel", "reduction", "reduction"]
      } ins(%1 : tensor<1x4x7x7xf32>) outs(%3 : tensor<1x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %add = arith.addf %in, %out : f32
      linalg.yield %add : f32
    } -> tensor<1x4xf32>

    %5 = linalg.generic {
        indexing_maps = [#map2d, #map2d],
        iterator_types = ["parallel", "parallel"]
      } ins(%4 : tensor<1x4xf32>) outs(%2 : tensor<1x4xf32>) {
    ^bb0(%in: f32, %out: f32):
      %div = arith.divf %in, %cst_count : f32
      linalg.yield %div : f32
    } -> tensor<1x4xf32>

    %6 = tensor.empty() : tensor<1x4xi8>
    %7 = linalg.generic {
        indexing_maps = [#map2d, #map2d],
        iterator_types = ["parallel", "parallel"]
      } ins(%5 : tensor<1x4xf32>) outs(%6 : tensor<1x4xi8>) {
    ^bb0(%in: f32, %out: i8):
      %div = arith.divf %in, %cst_out_scale : f32
      %rnd = math.roundeven %div : f32
      %add = arith.addf %rnd, %cst_zp_f32 : f32
      %clmp_lo = arith.maximumf %add, %cst_min : f32
      %clmp_hi = arith.minimumf %clmp_lo, %cst_max : f32
      %qi = arith.fptosi %clmp_hi : f32 to i8
      linalg.yield %qi : i8
    } -> tensor<1x4xi8>

    %8 = tensor.expand_shape %7 [[0], [1, 2, 3]] output_shape [1, 4, 1, 1]
      : tensor<1x4xi8> into tensor<1x4x1x1xi8>

    return %8 : tensor<1x4x1x1xi8>
  }
}
