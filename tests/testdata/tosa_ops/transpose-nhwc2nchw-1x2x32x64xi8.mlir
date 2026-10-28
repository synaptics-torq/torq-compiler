module {
  func.func @main(%arg0: tensor<1x2x32x64xi8>) -> tensor<1x64x2x32xi8> {
    %0 = tosa.transpose %arg0 {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x2x32x64xi8>) -> tensor<1x64x2x32xi8>
    return %0 : tensor<1x64x2x32xi8>
  }
}
