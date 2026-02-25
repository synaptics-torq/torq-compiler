module attributes {tf_saved_model.semantics} {
  func.func @main(%arg0: tensor<1x160x160x16xi8> {ml_program.identifier = "serving_default_images:0", tf_saved_model.index_path = ["images"]}) -> (tensor<1x160x160x16xi8> {ml_program.identifier = "PartitionedCall:0", tf_saved_model.index_path = ["output_0"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %206 = tosa.rescale %arg0 {double_round = true, input_zp = 5 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>} : (tensor<1x160x160x16xi8>) -> tensor<1x160x160x16xi32>
    %207 = tosa.rescale %arg0 {double_round = true, input_zp = -128 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>} : (tensor<1x160x160x16xi8>) -> tensor<1x160x160x16xi32>
    %208 = tosa.mul %206, %207 {shift = 0 : i8} : (tensor<1x160x160x16xi32>, tensor<1x160x160x16xi32>) -> tensor<1x160x160x16xi32>
    %209 = tosa.rescale %208 {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1111634816>, output_zp = -126 : i32, per_channel = false, scale32 = true, shift = array<i8: 37>} : (tensor<1x160x160x16xi32>) -> tensor<1x160x160x16xi8>
    return %209 : tensor<1x160x160x16xi8>
  }
}

