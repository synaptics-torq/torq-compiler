module {
  func.func @main(%arg0: tensor<128x49x2x2x3xi8>) -> tensor<49x2x2x3x128xi8> {
    %0 = tosa.transpose %arg0 {perms = array<i32: 1, 2, 3, 4, 0>} : (tensor<128x49x2x2x3xi8>) -> tensor<49x2x2x3x128xi8>
    return %0 : tensor<49x2x2x3x128xi8>
  }
}

