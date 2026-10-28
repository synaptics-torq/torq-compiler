module {
  func.func @main(%arg0: tensor<1x3x21x64xi8>) -> tensor<1x64x3x21xi8> {
    %0 = tosa.transpose %arg0 {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x3x21x64xi8>) -> tensor<1x64x3x21xi8>
    return %0 : tensor<1x64x3x21xi8>
  }
}
