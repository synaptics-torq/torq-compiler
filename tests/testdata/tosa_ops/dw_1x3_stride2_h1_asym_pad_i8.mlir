// Regression: DW KH=1xKW=3 stride-2 H=1 asym right-pad (EK Y-quadrant hang/mismatch).
module {
  func.func @main(%arg0: tensor<1x1x8x4xi8>) -> tensor<1x1x4x4xi8> {
    %bias = "tosa.const"() <{values = dense<0> : tensor<4xi32>}> : () -> tensor<4xi32>
    %weight = "tosa.const"() <{values = dense<[[[[1], [2], [3], [4]], [[1], [1], [1], [1]], [[-1], [-2], [-3], [-4]]]]> : tensor<1x3x4x1xi8>}> : () -> tensor<1x3x4x1xi8>
    %zp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %conv = tosa.depthwise_conv2d %arg0, %weight, %bias, %zp, %zp {
      acc_type = i32, dilation = array<i64: 1, 1>,
      pad = array<i64: 0, 0, 0, 1>, stride = array<i64: 2, 2>
    } : (tensor<1x1x8x4xi8>, tensor<1x3x4x1xi8>, tensor<4xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x1x4x4xi32>
    %mult = "tosa.const"() <{values = dense<1073741824> : tensor<4xi32>}> : () -> tensor<4xi32>
    %shift = "tosa.const"() <{values = dense<30> : tensor<4xi8>}> : () -> tensor<4xi8>
    %out_zp32 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out = tosa.rescale %conv, %mult, %shift, %out_zp32, %zp {
      input_unsigned = false, output_unsigned = false, per_channel = true,
      rounding_mode = DOUBLE_ROUND, scale32 = true
    } : (tensor<1x1x4x4xi32>, tensor<4xi32>, tensor<4xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x1x4x4xi8>
    return %out : tensor<1x1x4x4xi8>
  }
}
