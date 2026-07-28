// Int16 add of two full tensors: nothing to broadcast, so it stays fused. The
// multipliers still differ enough that a fixed scale shift quantizes the smaller one
// too coarsely, so this needs the grown shift and round-to-nearest weights.
// Extracted from NNNR3_0079_0.0960_int16x8 TFLite model (layer ADD_51).
module {
  func.func @main(%arg0: tensor<1x32xi16>) -> tensor<1x32xi16> {
    %0 = "tosa.const"() <{values = dense<43> : tensor<1xi8>}> : () -> tensor<1xi8>
    %1 = "tosa.const"() <{values = dense<1520509908> : tensor<1xi32>}> : () -> tensor<1xi32>
    %2 = "tosa.const"() <{values = dense<39> : tensor<1xi8>}> : () -> tensor<1xi8>
    %3 = "tosa.const"() <{values = dense<1908739182> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "tosa.const"() <{values = dense<15> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
    %5 = "tosa.const"() <{values = dense<[[7579, 32767, 25562, 10652, 29404, 9685, 19028, 1501, 28962, 9076, -5642, 17803, 3661, 28262, 8573, 22831, 10666, -4033, 19962, 26307, 23298, 5920, 23847, 28470, 13520, -14517, -3739, 3718, 30427, 19417, 5507, 8009]]> : tensor<1x32xi16>}> : () -> tensor<1x32xi16>
    %6 = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %7 = "tosa.const"() <{values = dense<16> : tensor<1xi8>}> : () -> tensor<1xi8>
    %8 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %9 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %10 = tosa.rescale %arg0, %6, %7, %8, %9 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x32xi16>, tensor<1xi32>, tensor<1xi8>, tensor<1xi16>, tensor<1xi32>) -> tensor<1x32xi32>
    %11 = tosa.cast %5 : (tensor<1x32xi16>) -> tensor<1x32xi32>
    %12 = tosa.logical_left_shift %11, %4 : (tensor<1x32xi32>, tensor<1x1xi32>) -> tensor<1x32xi32>
    %13 = tosa.rescale %12, %3, %2, %9, %9 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x32xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x32xi32>
    %14 = tosa.add %10, %13 : (tensor<1x32xi32>, tensor<1x32xi32>) -> tensor<1x32xi32>
    %15 = tosa.rescale %14, %1, %0, %9, %8 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x32xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi16>) -> tensor<1x32xi16>
    return %15 : tensor<1x32xi16>
  }
}
