module {
  func.func @main(%arg0: tensor<1x256x32xi8>) -> (tensor<1x16x32xi8>) {
    %0 = "tosa.const"() <{value = dense<[[200, 100, 30, 23, 45, 65, 76, 78, 90, 10, 22, 55, 15, 128, 222, 202]]> : tensor<1x16xi32>}> : () -> tensor<1x16xi32>
    %1 = tosa.gather %arg0, %0 : (tensor<1x256x32xi8>, tensor<1x16xi32>) -> tensor<1x16x32xi8>
    return %1 : tensor<1x16x32xi8>
  }
}

