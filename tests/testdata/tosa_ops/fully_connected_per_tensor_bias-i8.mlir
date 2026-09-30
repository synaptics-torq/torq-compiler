// TFLite FULLY_CONNECTED with a non-zero per-channel bias, a non-zero input zero point and a
// per-tensor rescale: the bias broadcast is rebuilt 2-D and the fused bias stays per output channel.
module {
  func.func @main(%arg0: tensor<4x16xi8>) -> tensor<4x8xi8> {
    %weights = "tosa.const"() <{values = dense<[[-93, 18, 89, 78, 68, -111, -62, -97, -1, 67, -12, -7, 39, -30, 74, -74], [-103, -3, -120, 101, 86, -28, -17, 28, 68, 69, -127, 51, -13, -59, 57, 78], [-69, 24, 114, -101, 103, -46, -120, -122, -121, 39, 11, -125, 113, 98, -30, 48], [-72, 121, -19, 58, -120, 8, -71, 68, -15, 113, -1, 14, -68, -39, -68, 46], [-71, 67, -10, 116, -53, 110, -122, -21, 87, 107, 15, 109, 37, -102, -80, 34], [127, 58, 93, -52, -97, 63, -42, 102, 57, 122, 55, 1, 112, 120, -19, 2], [85, 106, 44, -79, -50, -55, 23, 122, 98, 0, 89, 113, 2, -27, 23, 91], [-119, -5, -65, 63, 77, -24, -21, 43, -83, -34, 13, 98, 52, 71, 45, 61]]> : tensor<8x16xi8>}> : () -> tensor<8x16xi8>
    %bias = "tosa.const"() <{values = dense<[4556, -14334, 8767, 13320, -12927, -9272, 14140, 5772]> : tensor<8xi32>}> : () -> tensor<8xi32>
    %input_zp = "tosa.const"() <{values = dense<-5> : tensor<1xi8>}> : () -> tensor<1xi8>
    %weight_zp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %acc_zp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %output_zp = "tosa.const"() <{values = dense<3> : tensor<1xi8>}> : () -> tensor<1xi8>
    %multiplier = "tosa.const"() <{values = dense<1518500250> : tensor<1xi32>}> : () -> tensor<1xi32>
    %shift = "tosa.const"() <{values = dense<41> : tensor<1xi8>}> : () -> tensor<1xi8>
    %in_shape = tosa.const_shape {values = dense<[4, 1, 1, 16]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %w_shape = tosa.const_shape {values = dense<[8, 1, 1, 16]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %out_shape = tosa.const_shape {values = dense<[4, 8]> : tensor<2xindex>} : () -> !tosa.shape<2>
    %input = tosa.reshape %arg0, %in_shape : (tensor<4x16xi8>, !tosa.shape<4>) -> tensor<4x1x1x16xi8>
    %filter = tosa.reshape %weights, %w_shape : (tensor<8x16xi8>, !tosa.shape<4>) -> tensor<8x1x1x16xi8>
    %conv = tosa.conv2d %input, %filter, %bias, %input_zp, %weight_zp {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 1, 1>} : (tensor<4x1x1x16xi8>, tensor<8x1x1x16xi8>, tensor<8xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<4x1x1x8xi32>
    %rescaled = tosa.rescale %conv, %multiplier, %shift, %acc_zp, %output_zp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<4x1x1x8xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<4x1x1x8xi8>
    %out = tosa.reshape %rescaled, %out_shape : (tensor<4x1x1x8xi8>, !tosa.shape<2>) -> tensor<4x8xi8>
    return %out : tensor<4x8xi8>
  }
}
