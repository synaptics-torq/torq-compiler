module {
  func.func @main(%arg0: tensor<1x2x256x1xi8> {ml_program.identifier = "model_8/tf.tensor_scatter_nd_update/TensorScatterUpdate"}) -> (tensor<1x2x256x1xi8> {ml_program.identifier = "model_8/tf.tensor_scatter_nd_update_1/TensorScatterUpdate"}) {
    %0 = "tosa.const"() <{value = dense<127> : tensor<i8>}> : () -> tensor<i8>
    %1 = tosa.rescale %0 {double_round = true, input_zp = -128 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 11>} : (tensor<i8>) -> tensor<i32>
    %2 = tosa.rescale %arg0 {double_round = true, input_zp = -128 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 11>} : (tensor<1x2x256x1xi8>) -> tensor<1x2x256x1xi32>
    %3 = tosa.reshape %1 {new_shape = array<i64: 1, 1, 1, 1>} : (tensor<i32>) -> tensor<1x1x1x1xi32>
    %4 = tosa.sub %3, %2 : (tensor<1x1x1x1xi32>, tensor<1x2x256x1xi32>) -> tensor<1x2x256x1xi32>
    %5 = tosa.rescale %4 {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1073741824>, output_zp = -128 : i32, per_channel = false, scale32 = true, shift = array<i8: 49>} : (tensor<1x2x256x1xi32>) -> tensor<1x2x256x1xi8>
    return %5 : tensor<1x2x256x1xi8>
  }
}

