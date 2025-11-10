module {
  func.func @main(%arg0: tensor<1x8x4xi8>) -> (tensor<1x4x8xi8>) {
    %0 = "tosa.const"() <{value = dense<[0, 2, 1]> : tensor<3xi32>}> : () -> tensor<3xi32>
    %1 = tosa.transpose %arg0, %0 : (tensor<1x8x4xi8>, tensor<3xi32>) -> tensor<1x4x8xi8>
    return %1 : tensor<1x4x8xi8>
  }
}
