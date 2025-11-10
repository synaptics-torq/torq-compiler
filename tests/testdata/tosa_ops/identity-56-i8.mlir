module {
  func.func @main(%arg0: tensor<56xi8>) -> (tensor<56xi8>) {
    %0 = tosa.identity %arg0 : (tensor<56xi8>) -> tensor<56xi8>
    return %0 : tensor<56xi8>
  }
}
