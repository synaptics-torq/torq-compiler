func.func @main(%198 : tensor<1x21x1024xi16> ) -> tensor<1x21x1024xi32> {
    %199 = tosa.rescale %198 {double_round = false, input_zp = 0 : i32, multiplier = array<i32: 1073741824>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 30>} : (tensor<1x21x1024xi16>) -> tensor<1x21x1024xi32>
    return %199 : tensor<1x21x1024xi32>
}
