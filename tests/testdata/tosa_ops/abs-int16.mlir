module {
  func.func @main(%arg0: tensor<65x6xi16>) -> (tensor<65x6xi16>) {
    %0 = "tosa.abs"(%arg0) : (tensor<65x6xi16>) -> tensor<65x6xi16>
    return %0 : tensor<65x6xi16>
  }
}
