module {
  func.func @main(%191: tensor<1x21x1024xi32>) -> tensor<1x21x1024xi32> {
    %21 = "tosa.const"() <{value = dense<3> : tensor<1x1x1xi32>}> : () -> tensor<1x1x1xi32>
    %192 = tosa.arithmetic_right_shift %191, %21 {round = true} : (tensor<1x21x1024xi32>, tensor<1x1x1xi32>) -> tensor<1x21x1024xi32>
    return %192 : tensor<1x21x1024xi32>
  }
}
