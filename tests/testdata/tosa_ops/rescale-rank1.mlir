module {
  func.func @main(%740: tensor<1x17x2x2100xi8>) -> (tensor<1x17x2x2100xi32>) attributes {tf_saved_model.exported_names = ["serving_default"]} {

    %741 = tosa.rescale %740 {double_round = true, input_zp = 11 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>} : (tensor<1x17x2x2100xi8>) -> tensor<1x17x2x2100xi32>

    return %741 : tensor<1x17x2x2100xi32>
  }
}

