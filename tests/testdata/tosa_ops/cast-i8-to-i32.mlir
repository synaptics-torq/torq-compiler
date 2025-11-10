module {
  func.func @main(%arg0: tensor<1xi8>) -> (tensor<1xi32>) {
    %0 = tosa.cast %arg0 : (tensor<1xi8>) -> tensor<1xi32>
    return %0 : tensor<1xi32>
  }
}
