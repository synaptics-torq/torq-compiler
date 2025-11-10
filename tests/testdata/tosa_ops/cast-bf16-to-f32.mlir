module {
  func.func @main(%arg0: tensor<16x6xbf16>) -> (tensor<16x6xf32>) {
    %0 = tosa.cast %arg0 : (tensor<16x6xbf16>) -> tensor<16x6xf32>
    return %0 : tensor<16x6xf32>
  }
}
