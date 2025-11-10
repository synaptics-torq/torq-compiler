module {
  func.func @main(%arg0: tensor<1x5x16xi16>) -> (tensor<1x5x16xi32>) {
    %0 = tosa.cast %arg0 : (tensor<1x5x16xi16>) -> tensor<1x5x16xi32>
    return %0 : tensor<1x5x16xi32>
  }
}
