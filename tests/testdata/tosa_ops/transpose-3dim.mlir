module {
  func.func @main(%arg0: tensor<128x49x2xi8>) -> (tensor<49x2x128xi8>) {
    %0 = "tosa.const"() <{value = dense<[1, 2, 0]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %1 = tosa.transpose %arg0, %0 : (tensor<128x49x2xi8>, tensor<3xi32>) -> tensor<49x2x128xi8>
    return %1 : tensor<49x2x128xi8>
  }
}
