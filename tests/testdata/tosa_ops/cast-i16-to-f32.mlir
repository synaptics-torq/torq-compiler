module {
  func.func @main(%arg0: tensor<16x35xi16>) -> (tensor<16x35xf32>) {
    %0 = tosa.cast %arg0 : (tensor<16x35xi16>) -> tensor<16x35xf32>
    return %0 : tensor<16x35xf32>
  }
}
