module {
  func.func @main(%arg0: tensor<1x3x7x16xi32>) -> (tensor<1x3x7x16xi16>) {
    %0 = tosa.cast %arg0 : (tensor<1x3x7x16xi32>) -> tensor<1x3x7x16xi16>
    return %0 : tensor<1x3x7x16xi16>
  }
}
