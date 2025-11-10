module {
  func.func @main(%arg0: tensor<1x1x64x32x32xi8>) -> (tensor<1x1x32x64x32xi8>) {
    %0 = "tosa.const"() <{value = dense<[0, 1, 4, 2, 3]> : tensor<5xi32>}> : () -> tensor<5xi32>
    %1 = tosa.transpose %arg0, %0 : (tensor<1x1x64x32x32xi8>, tensor<5xi32>) -> tensor<1x1x32x64x32xi8>
    return %1 : tensor<1x1x32x64x32xi8>
  }
}
