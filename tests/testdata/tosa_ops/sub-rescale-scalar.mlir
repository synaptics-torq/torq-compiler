module {
  func.func @main(%arg0: tensor<1x320x320x1xi8>) -> (tensor<1x320x320x1xi8>) {
    %0 = "tosa.const"() <{value = dense<127> : tensor<1xi8>}> : () -> tensor<1xi8>
    %1 = "tosa.const"() <{value = dense<127> : tensor<i8>}> : () -> tensor<i8>
    %9 = tosa.rescale %arg0 {double_round = true, input_zp = -79 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 11>} : (tensor<1x320x320x1xi8>) -> tensor<1x320x320x1xi32>
    %10 = tosa.rescale %0 {double_round = true, input_zp = -128 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 10>} : (tensor<1xi8>) -> tensor<1xi32>
    %11 = tosa.rescale %10 {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1651596561>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 34>} : (tensor<1xi32>) -> tensor<1xi32>
    %12 = tosa.reshape %11 {new_shape = array<i64: 1, 1, 1, 1>} : (tensor<1xi32>) -> tensor<1x1x1x1xi32>
    %13 = tosa.sub %9, %12 : (tensor<1x320x320x1xi32>, tensor<1x1x1x1xi32>) -> tensor<1x320x320x1xi32>
    %14 = tosa.rescale %13 {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1329334528>, output_zp = -67 : i32, per_channel = false, scale32 = true, shift = array<i8: 49>} : (tensor<1x320x320x1xi32>) -> tensor<1x320x320x1xi8>
    return %14 : tensor<1x320x320x1xi8>
  }
}

