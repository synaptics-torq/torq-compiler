module {
  func.func @main(%arg0: tensor<1x21x1024xi32>) -> tensor<1x21x1024xi32> {
    %0 = "tosa.const"() <{values = dense<32767> : tensor<1x1x1xi32>}> : () -> tensor<1x1x1xi32>
    %1 = tosa.add %arg0, %0 : (tensor<1x21x1024xi32>, tensor<1x1x1xi32>) -> tensor<1x21x1024xi32>
    return %1 : tensor<1x21x1024xi32>
  }
}

