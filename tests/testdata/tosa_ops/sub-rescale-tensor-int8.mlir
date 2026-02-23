module {
  func.func @main(%arg0: tensor<1x16x128x1xi8> {ml_program.identifier = "model_11/tf.compat.v1.transpose_5/transpose;"}, %arg1: tensor<1x1x1x1xi8> {ml_program.identifier = "model_11/tf.math.reduce_mean/Mean"}) -> (tensor<1x16x128x1xi8> {ml_program.identifier = "model_11/tf.math.subtract/Sub"}) {
    %0 = tosa.rescale %arg0 {double_round = true, input_zp = 6 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 11>} : (tensor<1x16x128x1xi8>) -> tensor<1x16x128x1xi32>
    %1 = tosa.rescale %arg1 {double_round = true, input_zp = -128 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 10>} : (tensor<1x1x1x1xi8>) -> tensor<1x1x1x1xi32>
    %2 = tosa.rescale %1 {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 2039850423>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 39>} : (tensor<1x1x1x1xi32>) -> tensor<1x1x1x1xi32>
    %3 = tosa.sub %0, %2 : (tensor<1x16x128x1xi32>, tensor<1x1x1x1xi32>) -> tensor<1x16x128x1xi32>
    %4 = tosa.rescale %3 {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1077948431>, output_zp = 7 : i32, per_channel = false, scale32 = true, shift = array<i8: 49>} : (tensor<1x16x128x1xi32>) -> tensor<1x16x128x1xi8>
    return %4 : tensor<1x16x128x1xi8>
  }
}

