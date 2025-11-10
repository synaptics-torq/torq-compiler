module {
  func.func @main(%arg0: tensor<1x1001xi8>) -> (tensor<1x1001xui8>) {
    %0 = tosa.rescale %arg0 {double_round = false, input_zp = -70 : i32, multiplier = array<i32: 1073741824>, output_zp = 58 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>} : (tensor<1x1001xi8>) -> tensor<1x1001xui8>
    return %0 : tensor<1x1001xui8>
  }
}

