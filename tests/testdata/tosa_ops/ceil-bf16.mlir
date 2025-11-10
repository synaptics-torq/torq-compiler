module {
  func.func @main(%arg0: tensor<1x13x21x3xbf16>) -> tensor<1x13x21x3xbf16> {
    %0 = tosa.ceil %arg0 : (tensor<1x13x21x3xbf16>) -> tensor<1x13x21x3xbf16>
    return %0 : tensor<1x13x21x3xbf16>
  }
}
