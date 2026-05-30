module {
  func.func @main(%arg0: tensor<95x159xbf16>) -> tensor<159x95xbf16> {
    %0 = tosa.transpose %arg0 {perms = array<i32: 1, 0>} : (tensor<95x159xbf16>) -> tensor<159x95xbf16>
    return %0 : tensor<159x95xbf16>
  }
}

