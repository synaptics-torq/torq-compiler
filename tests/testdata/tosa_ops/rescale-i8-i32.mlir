module {
  func.func @main(%615: tensor<1x10x10x51xi8>) -> (tensor<1x10x10x51xi32>) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %617 = tosa.rescale %615 {double_round = true, input_zp = -10 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>} : (tensor<1x10x10x51xi8>) -> tensor<1x10x10x51xi32>
    return %617 : tensor<1x10x10x51xi32>
  }
}

