module {
  func.func @main(%arg0: tensor<1x64x32x32xi8>) -> tensor<1x32x32x64xi8> {
    %0 = tosa.transpose %arg0 {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x64x32x32xi8>) -> tensor<1x32x32x64xi8>
    return %0 : tensor<1x32x32x64xi8>
  }
}
