module {
  func.func @main(%arg0: tensor<1x56x56x256xi8>, %arg1: tensor<1x56x56x256xi8>) -> (tensor<1x56x56x256xi8>) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %128 = tosa.rescale %arg0 {double_round = true, input_zp = -68 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 10>} : (tensor<1x56x56x256xi8>) -> tensor<1x56x56x256xi32>
    %129 = tosa.rescale %128 {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1385091423>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 32>} : (tensor<1x56x56x256xi32>) -> tensor<1x56x56x256xi32>
    %130 = tosa.rescale %arg1 {double_round = true, input_zp = 11 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 11>} : (tensor<1x56x56x256xi8>) -> tensor<1x56x56x256xi32>
    %131 = tosa.sub %129, %130 : (tensor<1x56x56x256xi32>, tensor<1x56x56x256xi32>) -> tensor<1x56x56x256xi32>
    %132 = tosa.rescale %131 {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1086272925>, output_zp = -128 : i32, per_channel = false, scale32 = true, shift = array<i8: 48>} : (tensor<1x56x56x256xi32>) -> tensor<1x56x56x256xi8>
    return %132 : tensor<1x56x56x256xi8>
  }
}

