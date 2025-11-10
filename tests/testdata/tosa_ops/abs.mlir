module {
  func.func @main(%arg0: tensor<65xi8>) -> (tensor<65xi8>) {
    %0 = "tosa.abs"(%arg0) : (tensor<65xi8>) -> tensor<65xi8>
    return %0 : tensor<65xi8>
  }
}
