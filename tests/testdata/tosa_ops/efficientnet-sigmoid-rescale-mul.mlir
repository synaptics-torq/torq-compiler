module attributes {tf_saved_model.semantics} {
  func.func @main(%arg0: tensor<1x112x112x32xi8> {ml_program.identifier = "efficientnetb0_1/stem_conv_1/convolution;;efficientnetb0_1/stem_conv_pad_1/Pad1", tf_saved_model.index_path = ["input_164"]}, %arg1: tensor<1x112x112x32xi8> {ml_program.identifier = "efficientnetb0_1/stem_activation_1/Sigmoid", tf_saved_model.index_path = ["input_165"]}) -> (tensor<1x112x112x32xi8> {ml_program.identifier = "efficientnetb0_1/stem_activation_1/mul_1", tf_saved_model.index_path = ["output"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %6 = "tosa.const"() <{value = dense<0> : tensor<256xi8>}> : () -> tensor<256xi8>
    %5 = tosa.table %arg1, %6 : (tensor<1x112x112x32xi8>, tensor<256xi8>) -> tensor<1x112x112x32xi8>
    %0 = tosa.rescale %arg0 {double_round = true, input_zp = -26 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>} : (tensor<1x112x112x32xi8>) -> tensor<1x112x112x32xi32>
    %1 = tosa.rescale %5 {double_round = true, input_zp = -128 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>} : (tensor<1x112x112x32xi8>) -> tensor<1x112x112x32xi32>
    %2 = tosa.mul %0, %1 {shift = 0 : i8} : (tensor<1x112x112x32xi32>, tensor<1x112x112x32xi32>) -> tensor<1x112x112x32xi32>
    %3 = tosa.rescale %2 {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 2075768064>, output_zp = -26 : i32, per_channel = false, scale32 = true, shift = array<i8: 38>} : (tensor<1x112x112x32xi32>) -> tensor<1x112x112x32xi8>
    return %3 : tensor<1x112x112x32xi8>
  }
}

