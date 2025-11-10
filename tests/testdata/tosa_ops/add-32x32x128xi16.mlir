module {
  func.func @main(%arg0: tensor<1x32x32x128xi16>, %arg1: tensor<1x32x32x128xi16>) -> (tensor<1x32x32x128xi16>) {
    %7 = "tosa.const"() <{value = dense<15> : tensor<1x1x1x1xi32>}> : () -> tensor<1x1x1x1xi32>
    %151 = tosa.rescale %arg0 {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 16>} : (tensor<1x32x32x128xi16>) -> tensor<1x32x32x128xi32>
    %152 = tosa.cast %arg1 : (tensor<1x32x32x128xi16>) -> tensor<1x32x32x128xi32>
    %153 = tosa.logical_left_shift %152, %7 : (tensor<1x32x32x128xi32>, tensor<1x1x1x1xi32>) -> tensor<1x32x32x128xi32>
    %154 = tosa.rescale %153 {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1730653413>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 42>} : (tensor<1x32x32x128xi32>) -> tensor<1x32x32x128xi32>
    %155 = tosa.add %151, %154 : (tensor<1x32x32x128xi32>, tensor<1x32x32x128xi32>) -> tensor<1x32x32x128xi32>
    %156 = tosa.rescale %155 {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 44>} : (tensor<1x32x32x128xi32>) -> tensor<1x32x32x128xi16>
    return %156 : tensor<1x32x32x128xi16>
  }
}
