module {
  func.func @main(%arg0: tensor<95x159xi8>) -> tensor<159x95xi8> {
    %0 = tosa.transpose %arg0 {perms = array<i32: 1, 0>} : (tensor<95x159xi8>) -> tensor<159x95xi8>
    return %0 : tensor<159x95xi8>
  }
}

