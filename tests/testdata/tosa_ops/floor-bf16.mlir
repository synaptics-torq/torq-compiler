module {
  func.func @main(%arg0: tensor<1x13x21x3x2xbf16>) -> tensor<1x13x21x3x2xbf16> {
    %0 = tosa.floor %arg0 : (tensor<1x13x21x3x2xbf16>) -> tensor<1x13x21x3x2xbf16>
    return %0 : tensor<1x13x21x3x2xbf16>
  }
}
