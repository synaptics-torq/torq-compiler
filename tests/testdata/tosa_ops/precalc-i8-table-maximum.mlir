// A quantized leaky ReLU: two rescales of the same i8 activation joined by a maximum,
// then rescaled back to i8. There is no table anywhere in the input and the join is not a
// multiply, so this case is what shows the fold does not depend on either -- it depends
// only on the chain reading one i8 value and producing an i8 value.
module {
  func.func @main(%arg0: tensor<1x8x8x16xi8>) -> tensor<1x8x8x16xi8> {
    %alpha_mult = "tosa.const"() <{values = dense<1561768766> : tensor<1xi32>}> : () -> tensor<1xi32>
    %alpha_shift = "tosa.const"() <{values = dense<33> : tensor<1xi8>}> : () -> tensor<1xi8>
    %id_mult = "tosa.const"() <{values = dense<1952210929> : tensor<1xi32>}> : () -> tensor<1xi32>
    %id_shift = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>
    %out_mult = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_shift = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>
    %zero_i32 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %act_zp = "tosa.const"() <{values = dense<-1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %out_zp = "tosa.const"() <{values = dense<-105> : tensor<1xi8>}> : () -> tensor<1xi8>
    %zero_i8 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %unit_mult = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %unit_shift = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>

    %act = tosa.rescale %arg0, %unit_mult, %unit_shift, %zero_i8, %act_zp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x8x8x16xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x8x8x16xi8>
    // Both arms read the same activation; the scales are not identities.
    %neg = tosa.rescale %act, %alpha_mult, %alpha_shift, %act_zp, %zero_i32 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x8x8x16xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x8x8x16xi32>
    %pos = tosa.rescale %act, %id_mult, %id_shift, %act_zp, %zero_i32 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x8x8x16xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x8x8x16xi32>
    %joined = tosa.maximum %pos, %neg : (tensor<1x8x8x16xi32>, tensor<1x8x8x16xi32>) -> tensor<1x8x8x16xi32>
    %out = tosa.rescale %joined, %out_mult, %out_shift, %zero_i32, %out_zp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x8x8x16xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x8x8x16xi8>
    return %out : tensor<1x8x8x16xi8>
  }
}
