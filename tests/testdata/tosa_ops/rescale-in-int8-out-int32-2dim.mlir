module {
  func.func @main(%arg0: tensor<1x64xi8>) -> (tensor<1x64xi32>) {
    %0 = tosa.rescale %arg0 {double_round = true, input_zp = -128 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>} : (tensor<1x64xi8>) -> tensor<1x64xi32>
    return %0 : tensor<1x64xi32>
  }
}

