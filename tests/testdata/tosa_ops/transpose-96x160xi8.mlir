module {
  func.func @main(%arg0: tensor<96x160xi8>) -> tensor<160x96xi8> {
    %0 = tosa.transpose %arg0 {perms = array<i32: 1, 0>} : (tensor<96x160xi8>) -> tensor<160x96xi8>
    return %0 : tensor<160x96xi8>
  }
}

