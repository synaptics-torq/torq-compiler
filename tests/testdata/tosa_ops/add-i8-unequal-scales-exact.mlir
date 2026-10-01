// int8 add of two inputs with unequal scales, as TFLite emits it: each input is rescaled to i32
// (input 1 through a second 31-bit multiplier), added, and rescaled back to int8. The i16 add
// weights must hold the ratio of the two input scales precisely enough to round like TOSA, so the
// comparison is exact. Constants come from an add in the cofi_nss int8 model.
// TORQ_INT_TOL: 0
module {
  func.func @main(%a: tensor<1x1x64x64xi8>, %b: tensor<1x1x64x64xi8>) -> tensor<1x1x64x64xi8> {
    %m30 = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %mb = "tosa.const"() <{values = dense<1555757354> : tensor<1xi32>}> : () -> tensor<1xi32>
    %s11 = "tosa.const"() <{values = dense<11> : tensor<1xi8>}> : () -> tensor<1xi8>
    %s10 = "tosa.const"() <{values = dense<10> : tensor<1xi8>}> : () -> tensor<1xi8>
    %s32 = "tosa.const"() <{values = dense<32> : tensor<1xi8>}> : () -> tensor<1xi8>
    %s49 = "tosa.const"() <{values = dense<49> : tensor<1xi8>}> : () -> tensor<1xi8>
    %z16 = "tosa.const"() <{values = dense<16> : tensor<1xi8>}> : () -> tensor<1xi8>
    %z44 = "tosa.const"() <{values = dense<44> : tensor<1xi8>}> : () -> tensor<1xi8>
    %z0 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %0 = tosa.rescale %a, %m30, %s11, %z16, %z0 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x1x64x64xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x1x64x64xi32>
    %1 = tosa.rescale %b, %m30, %s10, %z44, %z0 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x1x64x64xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x1x64x64xi32>
    %2 = tosa.rescale %1, %mb, %s32, %z0, %z0 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x1x64x64xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x1x64x64xi32>
    %3 = tosa.add %0, %2 : (tensor<1x1x64x64xi32>, tensor<1x1x64x64xi32>) -> tensor<1x1x64x64xi32>
    %4 = tosa.rescale %3, %m30, %s49, %z0, %z16 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x1x64x64xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x1x64x64xi8>
    return %4 : tensor<1x1x64x64xi8>
  }
}
