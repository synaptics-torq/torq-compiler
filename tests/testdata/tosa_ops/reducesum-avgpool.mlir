module {
  func.func @main(%arg0: tensor<1x7x7x1280xi8>) -> (tensor<1x1280xi8>) {
    %301 = tosa.rescale %arg0 {double_round = true, input_zp = -128 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>} : (tensor<1x7x7x1280xi8>) -> tensor<1x7x7x1280xi32>
    %302 = tosa.reduce_sum %301 {axis = 1 : i32} : (tensor<1x7x7x1280xi32>) -> tensor<1x1x7x1280xi32>
    %303 = tosa.reduce_sum %302 {axis = 2 : i32} : (tensor<1x1x7x1280xi32>) -> tensor<1x1x1x1280xi32>
    %304 = tosa.rescale %303 {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 730269779>, output_zp = -128 : i32, per_channel = false, scale32 = true, shift = array<i8: 35>} : (tensor<1x1x1x1280xi32>) -> tensor<1x1x1x1280xi8>
    %305 = tosa.reshape %304 {new_shape = array<i64: -1, 1280>} : (tensor<1x1x1x1280xi8>) -> tensor<1x1280xi8>
    return %305 : tensor<1x1280xi8>
  }
}

