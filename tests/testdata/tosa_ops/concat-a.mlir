module {
  func.func @main(%arg0: tensor<1x21xi8>, %arg1: tensor<1x21xi8>) -> tensor<2x21xi8> {
    %1 = tosa.concat %arg0, %arg1 {axis = 0 : i32} : (tensor<1x21xi8>, tensor<1x21xi8>) -> tensor<2x21xi8>
    return %1 : tensor<2x21xi8>
  }
}
