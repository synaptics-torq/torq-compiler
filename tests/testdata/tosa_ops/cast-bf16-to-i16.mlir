module {
  func.func @main(%arg0: tensor<16x2xbf16>) -> (tensor<16x2xi16>) {
    %0 = tosa.cast %arg0 : (tensor<16x2xbf16>) -> tensor<16x2xi16>
    return %0 : tensor<16x2xi16>
  }
}
