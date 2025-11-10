module {
  func.func @main(%arg0: tensor<65x3x6xi32>) -> (tensor<65x3x6xi32>) {
    %0 = "tosa.abs"(%arg0) : (tensor<65x3x6xi32>) -> tensor<65x3x6xi32>
    return %0 : tensor<65x3x6xi32>
  }
}
