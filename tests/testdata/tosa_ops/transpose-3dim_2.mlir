module {
  func.func @main(%arg0: tensor<1x256x4xi8>) -> tensor<1x4x256xi8> {
    %0 = tosa.transpose %arg0 {perms = array<i32: 0, 2, 1>} : (tensor<1x256x4xi8>) -> tensor<1x4x256xi8>
    return %0 : tensor<1x4x256xi8>
  }
}

