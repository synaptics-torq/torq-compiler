module {
  func.func @main(%arg0: tensor<2x3x8x5x6x7xi8>) -> (tensor<2x3x8x5x6x7xi8>) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %133 = tosa.rescale %arg0 {double_round = true, input_zp = -2 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 10>} : (tensor<2x3x8x5x6x7xi8>) -> tensor<2x3x8x5x6x7xi32>
    %134 = tosa.rescale %133 {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1515257681>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 32>} : (tensor<2x3x8x5x6x7xi32>) -> tensor<2x3x8x5x6x7xi32>
    %135 = tosa.rescale %arg0 {double_round = true, input_zp = -3 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 11>} : (tensor<2x3x8x5x6x7xi8>) -> tensor<2x3x8x5x6x7xi32>
    %136 = tosa.add %134, %135 : (tensor<2x3x8x5x6x7xi32>, tensor<2x3x8x5x6x7xi32>) -> tensor<2x3x8x5x6x7xi32>
    %137 = tosa.rescale %136 {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1093202297>, output_zp = -2 : i32, per_channel = false, scale32 = true, shift = array<i8: 49>} : (tensor<2x3x8x5x6x7xi32>) -> tensor<2x3x8x5x6x7xi8>
    return %137 : tensor<2x3x8x5x6x7xi8>
  }
}

