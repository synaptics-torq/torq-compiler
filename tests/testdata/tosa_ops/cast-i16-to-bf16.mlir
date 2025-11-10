module {
  func.func @main(%arg0: tensor<16x8x3xi16>) -> (tensor<16x8x3xbf16>) {
    %0 = tosa.cast %arg0 : (tensor<16x8x3xi16>) -> tensor<16x8x3xbf16>
    return %0 : tensor<16x8x3xbf16>
  }
}
