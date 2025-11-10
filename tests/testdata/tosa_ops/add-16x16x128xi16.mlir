module {
  func.func @main(%arg0: tensor<1x16x16x128xi16>, %arg1: tensor<1x16x16x128xi16>) -> (tensor<1x16x16x128xi16>) {
    %7 = "tosa.const"() <{value = dense<15> : tensor<1x1x1x1xi32>}> : () -> tensor<1x1x1x1xi32>
    %129 = tosa.rescale %arg0 {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 16>} : (tensor<1x16x16x128xi16>) -> tensor<1x16x16x128xi32>
    %130 = tosa.cast %arg1 : (tensor<1x16x16x128xi16>) -> tensor<1x16x16x128xi32>
    %131 = tosa.logical_left_shift %130, %7 : (tensor<1x16x16x128xi32>, tensor<1x1x1x1xi32>) -> tensor<1x16x16x128xi32>
    %132 = tosa.rescale %131 {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1469854336>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 42>} : (tensor<1x16x16x128xi32>) -> tensor<1x16x16x128xi32>
    %133 = tosa.add %129, %132 : (tensor<1x16x16x128xi32>, tensor<1x16x16x128xi32>) -> tensor<1x16x16x128xi32>
    %134 = tosa.rescale %133 {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1074460101>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 44>} : (tensor<1x16x16x128xi32>) -> tensor<1x16x16x128xi16>
    return %134 : tensor<1x16x16x128xi16>
  }
}
