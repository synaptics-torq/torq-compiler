module {
  func.func @main(%arg0: tensor<1x56x56x24xi8> {ml_program.identifier = "serving_default_input_1:0", tf_saved_model.index_path = ["input_0"]}) -> (tensor<1x56x56x24xi8> {ml_program.identifier = "mobilenetv2_1.00_224/predictions/MatMul;mobilenetv2_1.00_224/predictions/BiasAdd", tf_saved_model.index_path = ["output"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = tosa.rescale %arg0 {double_round = true, input_zp = -3 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 11>} : (tensor<1x56x56x24xi8>) -> tensor<1x56x56x24xi32>
    %1 = tosa.add %0, %0 : (tensor<1x56x56x24xi32>, tensor<1x56x56x24xi32>) -> tensor<1x56x56x24xi32>
    %2 = tosa.rescale %1 {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1093202297>, output_zp = -2 : i32, per_channel = false, scale32 = true, shift = array<i8: 49>} : (tensor<1x56x56x24xi32>) -> tensor<1x56x56x24xi8>
    return %2 : tensor<1x56x56x24xi8>
  }
}

