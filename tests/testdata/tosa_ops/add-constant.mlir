module {
  func.func @main(%188: tensor<1x21x1024xi32>) -> (tensor<1x21x1024xi32>) {
    %22 = "tosa.const"() <{value = dense<32767> : tensor<1x1x1xi32>}> : () -> tensor<1x1x1xi32>
    %189 = tosa.add %188, %22 : (tensor<1x21x1024xi32>, tensor<1x1x1xi32>) -> tensor<1x21x1024xi32>
    return %189 : tensor<1x21x1024xi32>
  }
}
