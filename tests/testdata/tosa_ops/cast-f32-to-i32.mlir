module {
  func.func @main(%arg0: tensor<16x2xf32>) -> (tensor<16x2xi32>) {
    %0 = tosa.cast %arg0 : (tensor<16x2xf32>) -> tensor<16x2xi32>
    return %0 : tensor<16x2xi32>
  }
}
