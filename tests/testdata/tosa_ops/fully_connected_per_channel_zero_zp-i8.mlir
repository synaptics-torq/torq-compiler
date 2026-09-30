// TFLite FULLY_CONNECTED with zero zero points (a plain conv_2d_nhwc_hwcf, not the _q form), a
// non-zero bias and a per-channel rescale.
module {
  func.func @main(%arg0: tensor<4x16xi8>) -> tensor<4x8xi8> {
    %weights = "tosa.const"() <{values = dense<[[117, 93, 115, 90, -113, -104, -106, -35, 86, -84, 61, 80, 44, 91, -49, -63], [28, -73, 28, -118, 21, 47, -87, -17, 36, -27, 78, 58, 93, 127, 3, 116], [-32, 12, 112, -14, 1, -59, 103, -118, 95, -120, -34, -8, 111, -46, 105, -30], [-19, 101, 99, 7, -85, 16, -82, -67, -68, -121, -82, -44, -83, -93, 3, 3], [-35, 4, 45, 16, -81, 127, 101, -13, 76, -21, 61, 7, 105, 105, 68, -34], [75, 24, -37, -35, 124, 92, 119, -13, -86, 117, 66, -25, 56, 62, -9, 40], [8, -64, -2, -56, 109, 0, 1, 4, 85, 76, -37, 42, 99, -11, 103, 104], [-9, -38, 18, 58, 108, 15, 58, -11, -3, 41, -71, 113, -44, 81, 52, 86]]> : tensor<8x16xi8>}> : () -> tensor<8x16xi8>
    %bias = "tosa.const"() <{values = dense<[-9117, -2428, 11441, 287, -124, 13046, 16843, 13931]> : tensor<8xi32>}> : () -> tensor<8xi32>
    %input_zp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %weight_zp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %acc_zp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %output_zp = "tosa.const"() <{values = dense<3> : tensor<1xi8>}> : () -> tensor<1xi8>
    %multiplier = "tosa.const"() <{values = dense<[1947060140, 1743422415, 1520028602, 2123660442, 1861005555, 1235609193, 1807019266, 1091827547]> : tensor<8xi32>}> : () -> tensor<8xi32>
    %shift = "tosa.const"() <{values = dense<[40, 42, 40, 40, 42, 42, 40, 41]> : tensor<8xi8>}> : () -> tensor<8xi8>
    %in_shape = tosa.const_shape {values = dense<[4, 1, 1, 16]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %w_shape = tosa.const_shape {values = dense<[8, 1, 1, 16]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %out_shape = tosa.const_shape {values = dense<[4, 8]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %input = tosa.reshape %arg0, %in_shape : (tensor<4x16xi8>, !tosa.shape<4>) -> tensor<4x1x1x16xi8>
    %filter = tosa.reshape %weights, %w_shape : (tensor<8x16xi8>, !tosa.shape<4>) -> tensor<8x1x1x16xi8>
    %conv = tosa.conv2d %input, %filter, %bias, %input_zp, %weight_zp {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<4x1x1x16xi8>, tensor<8x1x1x16xi8>, tensor<8xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<4x1x1x8xi32>
    %rescaled = tosa.rescale %conv, %multiplier, %shift, %acc_zp, %output_zp {input_unsigned = false, output_unsigned = false, per_channel = true, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<4x1x1x8xi32>, tensor<8xi32>, tensor<8xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<4x1x1x8xi8>
    %out = tosa.reshape %rescaled, %out_shape : (tensor<4x1x1x8xi8>, !tosa.shape<2>) -> tensor<4x8xi8>
    return %out : tensor<4x8xi8>
  }
}
