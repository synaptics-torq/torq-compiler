module {
  func.func @main(%arg0: tensor<1x256x1xi8>) -> (tensor<1x256x1xi8>) {
    %0 = "tosa.const"() <{value = dense<[[200, 100, 30, 23, 45, 65, 76, 78]]> : tensor<1x8xi32>}> : () -> tensor<1x8xi32>
    %1 = "tosa.const"() <{value = dense<[[[1], [5], [9], [13], [17], [21], [25], [29]]]> : tensor<1x8x1xi8>}> : () -> tensor<1x8x1xi8>
    %2 = tosa.scatter %arg0, %0, %1: (tensor<1x256x1xi8>, tensor<1x8xi32>, tensor<1x8x1xi8>) -> tensor<1x256x1xi8>
    return %2 : tensor<1x256x1xi8>
  }
}