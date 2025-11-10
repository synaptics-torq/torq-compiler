module {
  func.func @main(%arg0: tensor<128x49xi8>) -> (tensor<49x128xi8>) {
    %0 = "tosa.const"() <{value = dense<[1, 0]> : tensor<2xi32>}> : () -> tensor<2xi32>
    %1 = tosa.transpose %arg0, %0 : (tensor<128x49xi8>, tensor<2xi32>) -> tensor<49x128xi8>
    return %1 : tensor<49x128xi8>
  }
}
