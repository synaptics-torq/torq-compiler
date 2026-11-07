// Quantized mul: the input zero point -127 folds as an offset of 127 and the zero point 0 folds as a plain sign extension, so the mul is one slice task.
module {
  func.func @main(%arg0: tensor<2x5x70xi8>, %arg1: tensor<2x5x70xi8>) -> tensor<2x5x70xi8> {
    %mult_up = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %shift_up = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>
    %zp_a = "tosa.const"() <{values = dense<-127> : tensor<1xi8>}> : () -> tensor<1xi8>
    %zp_b = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %zp_i32 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %mul_shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %mult_out = "tosa.const"() <{values = dense<1227697152> : tensor<1xi32>}> : () -> tensor<1xi32>
    %shift_out = "tosa.const"() <{values = dense<38> : tensor<1xi8>}> : () -> tensor<1xi8>
    %zp_out = "tosa.const"() <{values = dense<-27> : tensor<1xi8>}> : () -> tensor<1xi8>
    %a = tosa.rescale %arg0, %mult_up, %shift_up, %zp_a, %zp_i32 scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<2x5x70xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<2x5x70xi32>
    %b = tosa.rescale %arg1, %mult_up, %shift_up, %zp_b, %zp_i32 scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<2x5x70xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<2x5x70xi32>
    %m = tosa.mul %a, %b, %mul_shift : (tensor<2x5x70xi32>, tensor<2x5x70xi32>, tensor<1xi8>) -> tensor<2x5x70xi32>
    %r = tosa.rescale %m, %mult_out, %shift_out, %zp_i32, %zp_out scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<2x5x70xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<2x5x70xi8>
    return %r : tensor<2x5x70xi8>
  }
}
