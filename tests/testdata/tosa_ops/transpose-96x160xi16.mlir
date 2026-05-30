module {
  func.func @main(%arg0: tensor<96x160xi16>) -> tensor<160x96xi16> {
    %0 = tosa.transpose %arg0 {perms = array<i32: 1, 0>} : (tensor<96x160xi16>) -> tensor<160x96xi16>
    return %0 : tensor<160x96xi16>
  }
}

