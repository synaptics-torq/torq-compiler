module {
  func.func @main(%194: tensor<1x21x1xi32>) -> (tensor<1x21x1xi32>) {
    %16 = "tosa.const"() <{value = dense<31> : tensor<1x1x1xi32>}> : () -> tensor<1x1x1xi32>
    %204 = tosa.sub %16, %194 : (tensor<1x1x1xi32>, tensor<1x21x1xi32>) -> tensor<1x21x1xi32>
    return %204 : tensor<1x21x1xi32>
  }
}
