module {
  func.func @main(%arg0: tensor<1x56x56x3xi32>) -> (tensor<1x56x56x3xi8>) {
    %241 = tosa.rescale %arg0 {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1106700928>, output_zp = 125 : i32, per_channel = false, scale32 = true, shift = array<i8: 45>} : (tensor<1x56x56x3xi32>) -> tensor<1x56x56x3xi8>
    return %241 : tensor<1x56x56x3xi8>
  }
}
