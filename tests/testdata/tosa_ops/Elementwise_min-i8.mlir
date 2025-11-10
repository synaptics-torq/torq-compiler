module {
  func.func @main(%arg0: tensor<4x64xi8>, %arg1: tensor<4x64xi8>) -> tensor<4x64xi8> {
    %0 = tosa.minimum %arg0, %arg1 : (tensor<4x64xi8>, tensor<4x64xi8>) -> tensor<4x64xi8>
    return %0 : tensor<4x64xi8>
  }
}