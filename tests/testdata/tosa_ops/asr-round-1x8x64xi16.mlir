// Rounding right shift of i16 by a constant: the shift amount must keep the i16 element type.
module {
  func.func @main(%arg0: tensor<1x8x64xi16>) -> tensor<1x8x64xi16> {
    %s = "tosa.const"() <{values = dense<3> : tensor<1x1x1xi16>}> : () -> tensor<1x1x1xi16>
    %r = tosa.arithmetic_right_shift %arg0, %s {round = true} : (tensor<1x8x64xi16>, tensor<1x1x1xi16>) -> tensor<1x8x64xi16>
    return %r : tensor<1x8x64xi16>
  }
}
