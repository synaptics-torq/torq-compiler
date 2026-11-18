// Regression: a unit-scale signed i8 -> UINT8 output rescale (output_zp - input_zp
// == 128) must lower NSS-only. Every byte maps exactly to x + 128, i.e. x XOR 0x80.
module {
  func.func @main(%arg0: tensor<1x8x8x4xi8>) -> tensor<1x8x8x4xui8> {
    %multiplier = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %shift = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>
    %izp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %ozp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %result = tosa.rescale %arg0, %multiplier, %shift, %izp, %ozp scale32(true) rounding_mode<SINGLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(true) : (tensor<1x8x8x4xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x8x8x4xui8>
    return %result : tensor<1x8x8x4xui8>
  }
}
