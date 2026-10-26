// Quantized add whose first operand is the i32 result of a mul, not a rescale of an i8 tensor.
// The fused add multiplies each operand by an i16 weight, and the ALU cannot do that for an i32
// operand. The add must stay an i32 add followed by a separate requantization. Before the fix
// the fused lowering took the i32 operand and failed with "Add input element types must match".
module {
  func.func @main(%arg0: tensor<1x8x16xi8>, %arg1: tensor<1x8x16xi8>, %arg2: tensor<1x8x16xi8>) -> tensor<1x8x16xi8> {
    %mult_one = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %shift_one = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>
    %shift_up = "tosa.const"() <{values = dense<20> : tensor<1xi8>}> : () -> tensor<1xi8>
    %zp_a = "tosa.const"() <{values = dense<-3> : tensor<1xi8>}> : () -> tensor<1xi8>
    %zp_b = "tosa.const"() <{values = dense<5> : tensor<1xi8>}> : () -> tensor<1xi8>
    %zp_c = "tosa.const"() <{values = dense<-7> : tensor<1xi8>}> : () -> tensor<1xi8>
    %zp_i32 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %mul_shift = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %mult_out = "tosa.const"() <{values = dense<1717986918> : tensor<1xi32>}> : () -> tensor<1xi32>
    %shift_out = "tosa.const"() <{values = dense<41> : tensor<1xi8>}> : () -> tensor<1xi8>
    %zp_out = "tosa.const"() <{values = dense<4> : tensor<1xi8>}> : () -> tensor<1xi8>
    %a = tosa.rescale %arg0, %mult_one, %shift_one, %zp_a, %zp_i32 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x8x16xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x8x16xi32>
    %b = tosa.rescale %arg1, %mult_one, %shift_one, %zp_b, %zp_i32 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x8x16xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x8x16xi32>
    %m = tosa.mul %a, %b, %mul_shift : (tensor<1x8x16xi32>, tensor<1x8x16xi32>, tensor<1xi8>) -> tensor<1x8x16xi32>
    %c = tosa.rescale %arg2, %mult_one, %shift_up, %zp_c, %zp_i32 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x8x16xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x8x16xi32>
    %s = tosa.add %m, %c : (tensor<1x8x16xi32>, tensor<1x8x16xi32>) -> tensor<1x8x16xi32>
    %r = tosa.rescale %s, %mult_out, %shift_out, %zp_i32, %zp_out {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x8x16xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x8x16xi8>
    return %r : tensor<1x8x16xi8>
  }
}
