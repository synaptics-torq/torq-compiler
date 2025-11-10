module {
  func.func @main(%arg0: tensor<1x3x5x16xi16>) -> (tensor<1x3x5x16xi8>) {
    %0 = tosa.cast %arg0 : (tensor<1x3x5x16xi16>) -> tensor<1x3x5x16xi8>
    return %0 : tensor<1x3x5x16xi8>
  }
}
