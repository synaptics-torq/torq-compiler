module {
  func.func @main(%194: tensor<1x21x1xi32>) -> (tensor<1x21x1xi32>) {
    %20 = "tosa.const"() <{value = dense<1> : tensor<1x1x1xi32>}> : () -> tensor<1x1x1xi32>
    %195 = tosa.sub %194, %20 : (tensor<1x21x1xi32>, tensor<1x1x1xi32>) -> tensor<1x21x1xi32>
    return %195 : tensor<1x21x1xi32>
  }
}
