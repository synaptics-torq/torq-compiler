func.func @main(%201 : tensor<1x21x1024xi32>) -> tensor<1x21x1024xi32> {
    %202 = tosa.rescale %201 {double_round = true, input_zp = 0 : i32, multiplier = array<i32: 1758038019>, output_zp = 0 : i32, per_channel = false, scale32 = true, shift = array<i8: 28>} : (tensor<1x21x1024xi32>) -> tensor<1x21x1024xi32>
    return %202 : tensor<1x21x1024xi32>
}
