// disable this test because padding issue, will enable it after tensor encoding is complete

module {
  func.func @main(%arg0: tensor<1x68x68x7xi8>, %arg1: tensor<1x68x68x7xi8>) -> (tensor<1x68x68x7xi32>) {
    %0 = tosa.rescale %arg1 {double_round = true, input_zp = -128 : i32, multiplier = array<i32: 1073741824>, output_zp = -128 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>} : (tensor<1x68x68x7xi8>) -> tensor<1x68x68x7xi8>
    %1 = tosa.rescale %arg0 {double_round = true, input_zp = -128 : i32, multiplier = array<i32: 1073741824>, output_zp = -128 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>} : (tensor<1x68x68x7xi8>) -> tensor<1x68x68x7xi8>
    %2 = tosa.mul %0, %1 {shift = 0 : i8} : (tensor<1x68x68x7xi8>, tensor<1x68x68x7xi8>) -> tensor<1x68x68x7xi32>
    return %2 : tensor<1x68x68x7xi32>
  }
}

