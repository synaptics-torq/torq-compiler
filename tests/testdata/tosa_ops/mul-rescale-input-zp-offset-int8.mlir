// Quantized mul: the input zero point 127 folds into the mul as an offset of -127; the zero point -128 does not fit an i8 offset and keeps its rescale. 37 is not a multiple of the vector size.
module {
  func.func @main(%arg0: tensor<1x13x37xi8>, %arg1: tensor<1x13x37xi8>) -> tensor<1x13x37xi8> {
    %mult_up = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %shift_up = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>
    %zp_a = "tosa.const"() <{values = dense<127> : tensor<1xi8>}> : () -> tensor<1xi8>
    %zp_b = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %zp_i32 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %mul_shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %mult_out = "tosa.const"() <{values = dense<1083530240> : tensor<1xi32>}> : () -> tensor<1xi32>
    %shift_out = "tosa.const"() <{values = dense<38> : tensor<1xi8>}> : () -> tensor<1xi8>
    %zp_out = "tosa.const"() <{values = dense<3> : tensor<1xi8>}> : () -> tensor<1xi8>
    %a = tosa.rescale %arg0, %mult_up, %shift_up, %zp_a, %zp_i32 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x13x37xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x13x37xi32>
    %b = tosa.rescale %arg1, %mult_up, %shift_up, %zp_b, %zp_i32 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x13x37xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x13x37xi32>
    %m = tosa.mul %a, %b, %mul_shift : (tensor<1x13x37xi32>, tensor<1x13x37xi32>, tensor<1xi8>) -> tensor<1x13x37xi32>
    %r = tosa.rescale %m, %mult_out, %shift_out, %zp_i32, %zp_out {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x13x37xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x13x37xi8>
    return %r : tensor<1x13x37xi8>
  }
}
