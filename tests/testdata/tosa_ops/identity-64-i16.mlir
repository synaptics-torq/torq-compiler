module {
  func.func @main(%arg0: tensor<64xi16>) -> (tensor<64xi16>) {
    %0 = tosa.identity %arg0 : (tensor<64xi16>) -> tensor<64xi16>
    return %0 : tensor<64xi16>
  }
}