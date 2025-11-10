module {
  func.func @main(%arg0: tensor<16x3x4xbf16>) -> (tensor<16x3x4xi32>) {
    %0 = tosa.cast %arg0 : (tensor<16x3x4xbf16>) -> tensor<16x3x4xi32>
    return %0 : tensor<16x3x4xi32>
  }
}
