// Quantized mul with two non-zero input zero points, -3 and 5: only one input can carry the i8 offset,
// so the zero point -3 folds into the mul and the zero point 5 keeps its rescale.
module {
  func.func @main(%arg0: tensor<3x20x24xi8>, %arg1: tensor<3x20x24xi8>) -> tensor<3x20x24xi8> {
    %mult_up = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %shift_up = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>
    %zp_a = "tosa.const"() <{values = dense<-3> : tensor<1xi8>}> : () -> tensor<1xi8>
    %zp_b = "tosa.const"() <{values = dense<5> : tensor<1xi8>}> : () -> tensor<1xi8>
    %zp_i32 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %mul_shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %mult_out = "tosa.const"() <{values = dense<1083530240> : tensor<1xi32>}> : () -> tensor<1xi32>
    %shift_out = "tosa.const"() <{values = dense<38> : tensor<1xi8>}> : () -> tensor<1xi8>
    %zp_out = "tosa.const"() <{values = dense<3> : tensor<1xi8>}> : () -> tensor<1xi8>
    %a = tosa.rescale %arg0, %mult_up, %shift_up, %zp_a, %zp_i32 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<3x20x24xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<3x20x24xi32>
    %b = tosa.rescale %arg1, %mult_up, %shift_up, %zp_b, %zp_i32 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<3x20x24xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<3x20x24xi32>
    %m = tosa.mul %a, %b, %mul_shift : (tensor<3x20x24xi32>, tensor<3x20x24xi32>, tensor<1xi8>) -> tensor<3x20x24xi32>
    %r = tosa.rescale %m, %mult_out, %shift_out, %zp_i32, %zp_out {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<3x20x24xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<3x20x24xi8>
    return %r : tensor<3x20x24xi8>
  }
}
