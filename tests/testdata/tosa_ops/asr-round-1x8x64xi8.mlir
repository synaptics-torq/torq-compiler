// Rounding right shift of i8 by a constant: the shift amount must keep the i8 element type.
module {
  func.func @main(%arg0: tensor<1x8x64xi8>) -> tensor<1x8x64xi8> {
    %s = "tosa.const"() <{values = dense<3> : tensor<1x1x1xi8>}> : () -> tensor<1x1x1xi8>
    %r = tosa.arithmetic_right_shift %arg0, %s round(true) : (tensor<1x8x64xi8>, tensor<1x1x1xi8>) -> tensor<1x8x64xi8>
    return %r : tensor<1x8x64xi8>
  }
}
