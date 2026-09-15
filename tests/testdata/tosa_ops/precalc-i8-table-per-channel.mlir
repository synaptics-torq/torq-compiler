// Negative case: the multiply's other operand is a per-channel weight broadcast over the
// channel axis, the way a quantized PReLU legalizes. The result is a function of the
// activation and the channel, which no 256-entry table covers, so the fold has to leave
// this graph alone and the numbers still have to match the llvm-cpu reference.
module {
  func.func @main(%arg0: tensor<1x8x8x16xi8>) -> tensor<1x8x8x16xi8> {
    %w = "tosa.const"() <{values = dense<[[[[-30,-23,-16,-9,-2,5,12,19,26,-28,-21,-14,-7,0,7,14]]]]> : tensor<1x1x1x16xi8>}> : () -> tensor<1x1x1x16xi8>
    %out_mult = "tosa.const"() <{values = dense<1854308352> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_shift = "tosa.const"() <{values = dense<38> : tensor<1xi8>}> : () -> tensor<1xi8>
    %zero_i32 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %zero_i8 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %act_zp = "tosa.const"() <{values = dense<-18> : tensor<1xi8>}> : () -> tensor<1xi8>
    %w_zp = "tosa.const"() <{values = dense<-3> : tensor<1xi8>}> : () -> tensor<1xi8>
    %out_zp = "tosa.const"() <{values = dense<-123> : tensor<1xi8>}> : () -> tensor<1xi8>
    %unit_mult = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %unit_shift = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>

    %act = tosa.rescale %arg0, %unit_mult, %unit_shift, %zero_i8, %act_zp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x8x8x16xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x8x8x16xi8>
    %direct = tosa.rescale %act, %unit_mult, %unit_shift, %act_zp, %zero_i32 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x8x8x16xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x8x8x16xi32>
    // Per-channel, so the product is a function of the activation and the channel.
    %wide_w = tosa.rescale %w, %unit_mult, %unit_shift, %w_zp, %zero_i32 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x1x1x16xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x1x1x16xi32>
    %prod = tosa.mul %direct, %wide_w, %zero_i8 : (tensor<1x8x8x16xi32>, tensor<1x1x1x16xi32>, tensor<1xi8>) -> tensor<1x8x8x16xi32>
    %out = tosa.rescale %prod, %out_mult, %out_shift, %zero_i32, %out_zp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x8x8x16xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x8x8x16xi8>
    return %out : tensor<1x8x8x16xi8>
  }
}
