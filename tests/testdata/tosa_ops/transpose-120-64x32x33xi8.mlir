module {
  func.func @main(%arg0: tensor<64x32x33xi8>) -> tensor<32x33x64xi8> {
    %0 = tosa.transpose %arg0 {perms = array<i32: 1, 2, 0>} : (tensor<64x32x33xi8>) -> tensor<32x33x64xi8>
    return %0 : tensor<32x33x64xi8>
  }
}
