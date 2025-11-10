module {
  func.func @main(%arg0: tensor<16xi32>) -> (tensor<16xi8>) {
    %0 = tosa.cast %arg0 : (tensor<16xi32>) -> tensor<16xi8>
    return %0 : tensor<16xi8>
  }
}
