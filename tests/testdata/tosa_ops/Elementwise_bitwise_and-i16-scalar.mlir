module {
  func.func @main(%arg0: tensor<1x1024x64xi16>) -> tensor<1x1024x64xi16> {
    %0 = "tosa.const"() <{value = dense<255> : tensor<1xi16>}> : () -> tensor<1xi16>
    %1 = tosa.bitwise_and %0, %arg0 : (tensor<1xi16>, tensor<1x1024x64xi16>) -> tensor<1x1024x64xi16>
    return %1 : tensor<1x1024x64xi16>
  }
}
