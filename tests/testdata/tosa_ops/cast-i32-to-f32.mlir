module {
  func.func @main(%arg0: tensor<16x7xi32>) -> (tensor<16x7xf32>) {
    %0 = tosa.cast %arg0 : (tensor<16x7xi32>) -> tensor<16x7xf32>
    return %0 : tensor<16x7xf32>
  }
}
