// as currently hw not ready for int16 input
// it is not necessary rescale from int8 to int8, we use this test just to check if rescale->FMAOp -> op is working
// the valid case would be rescale input from int8 to int16, then send int16 to op
// we will enable it after hw is ready for int16 input

module {
  func.func @main(%arg0: tensor<1x64x64x7xi8>, %arg1: tensor<1x64x64x7xi8>) -> (tensor<1x64x64x7xi32>) {
    %0 = tosa.rescale %arg1 {double_round = true, input_zp = -128 : i32, multiplier = array<i32: 1073741824>, output_zp = -128 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>} : (tensor<1x64x64x7xi8>) -> tensor<1x64x64x7xi8>
    %1 = tosa.rescale %arg0 {double_round = true, input_zp = -128 : i32, multiplier = array<i32: 1073741824>, output_zp = -128 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>} : (tensor<1x64x64x7xi8>) -> tensor<1x64x64x7xi8>
    %2 = tosa.mul %0, %1 {shift = 0 : i8} : (tensor<1x64x64x7xi8>, tensor<1x64x64x7xi8>) -> tensor<1x64x64x7xi32>
    return %2 : tensor<1x64x64x7xi32>
  }
}

