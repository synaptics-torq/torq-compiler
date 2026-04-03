module {
  func.func @main(%arg0: tensor<1x21x1xi32>) -> tensor<1x21x1xi32> {
    %0 = "tosa.const"() <{values = dense<31> : tensor<1x1x1xi32>}> : () -> tensor<1x1x1xi32>
    %1 = tosa.sub %0, %arg0 : (tensor<1x1x1xi32>, tensor<1x21x1xi32>) -> tensor<1x21x1xi32>
    return %1 : tensor<1x21x1xi32>
  }
}

