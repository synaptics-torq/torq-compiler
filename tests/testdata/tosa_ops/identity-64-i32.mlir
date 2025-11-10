module {
  func.func @main(%arg0: tensor<64xi32>) -> (tensor<64xi32>) {
    %0 = tosa.identity %arg0 : (tensor<64xi32>) -> tensor<64xi32>
    return %0 : tensor<64xi32>
  }
}