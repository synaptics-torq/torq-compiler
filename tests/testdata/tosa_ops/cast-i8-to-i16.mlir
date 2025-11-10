module {
  func.func @main(%arg0: tensor<16xi16>) -> (tensor<16xi32>) {
    %0 = tosa.cast %arg0 : (tensor<16xi16>) -> tensor<16xi32>
    return %0 : tensor<16xi32>
  }
}
