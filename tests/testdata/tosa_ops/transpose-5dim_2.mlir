module {
  func.func @main(%arg0: tensor<1x1x64x2x36xi8>) -> (tensor<1x1x2x36x64xi8>) {
    %0 = "tosa.const"() <{value = dense<[0, 1, 3, 4, 2]> : tensor<5xi32>}> : () -> tensor<5xi32>
    %1 = tosa.transpose %arg0, %0 : (tensor<1x1x64x2x36xi8>, tensor<5xi32>) -> tensor<1x1x2x36x64xi8>
    return %1 : tensor<1x1x2x36x64xi8>
  }
}
