module {
  func.func @main(%arg0: tensor<4x64xi8>, %arg1: tensor<4x64xi8>) -> tensor<4x64xi1> {
    %0 = tosa.equal %arg0, %arg1 : (tensor<4x64xi8>, tensor<4x64xi8>) -> tensor<4x64xi1>
    return %0 : tensor<4x64xi1>
  }
}
