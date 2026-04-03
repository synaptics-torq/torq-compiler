module {
  func.func @main(%arg0: tensor<1x1280x7x7xi8>) -> tensor<1x7x7x1280xi8> {
    %0 = tosa.transpose %arg0 {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x1280x7x7xi8>) -> tensor<1x7x7x1280xi8>
    return %0 : tensor<1x7x7x1280xi8>
  }
}

