module {
  func.func @main(%arg0: tensor<4x64xi16>, %arg1: tensor<4x64xi16>) -> tensor<4x64xi16> {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<4x64xi16>, tensor<4x64xi16>) -> tensor<4x64xi16>
    return %0 : tensor<4x64xi16>
  }
}