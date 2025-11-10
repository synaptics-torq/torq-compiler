module {
  func.func @main(%619: tensor<1x10x10x51xi32>) -> (tensor<1x10x10x51xi8>) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %620 = tosa.rescale %619 {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1919293440>, output_zp = -118 : i32, per_channel = false, scale32 = true, shift = array<i8: 38>} : (tensor<1x10x10x51xi32>) -> tensor<1x10x10x51xi8>
    return %620 : tensor<1x10x10x51xi8>
  }
}

