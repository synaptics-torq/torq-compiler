// Int16 per-channel bias add. The two operands need very different multipliers, so
// approximating both with i16 ACT weights under one shared shift loses an LSB: this
// must take the unfused lowering, which rescales exactly with 32-bit multipliers.
// Extracted from NNNR3_0079_0.0960_int16x8 TFLite model (layer ADD_619).
module {
  func.func @main(%arg0: tensor<1x1x64x48xi16>) -> tensor<1x1x64x48xi16> {
    %0 = "tosa.const"() <{values = dense<44> : tensor<1xi8>}> : () -> tensor<1xi8>
    %1 = "tosa.const"() <{values = dense<1879508052> : tensor<1xi32>}> : () -> tensor<1xi32>
    %2 = tosa.const_shape  {values = dense<[1, 1, 1, 48]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %3 = "tosa.const"() <{values = dense<37> : tensor<1xi8>}> : () -> tensor<1xi8>
    %4 = "tosa.const"() <{values = dense<1258321386> : tensor<1xi32>}> : () -> tensor<1xi32>
    %5 = "tosa.const"() <{values = dense<15> : tensor<1xi32>}> : () -> tensor<1xi32>
    %6 = "tosa.const"() <{values = dense<[19852, -288, 12525, -3043, -7380, 22726, 3596, 5445, -32223, 6262, 19806, -6614, 9890, 8030, 14090, -10379, 12245, 11702, 3053, 7278, 7874, -977, -1351, 9429, 13203, 13031, -7213, 3340, 3080, 20564, 11770, 4953, 22398, 9849, 847, 5698, 2956, -4235, 24906, 3728, 308, 11040, -7091, 6124, 25440, -3804, -32767, -10966]> : tensor<48xi16>}> : () -> tensor<48xi16>
    %7 = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %8 = "tosa.const"() <{values = dense<16> : tensor<1xi8>}> : () -> tensor<1xi8>
    %9 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %10 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %11 = tosa.rescale %arg0, %7, %8, %9, %10 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x1x64x48xi16>, tensor<1xi32>, tensor<1xi8>, tensor<1xi16>, tensor<1xi32>) -> tensor<1x1x64x48xi32>
    %12 = tosa.cast %6 : (tensor<48xi16>) -> tensor<48xi32>
    %13 = tosa.logical_left_shift %12, %5 : (tensor<48xi32>, tensor<1xi32>) -> tensor<48xi32>
    %14 = tosa.rescale %13, %4, %3, %10, %10 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<48xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<48xi32>
    %15 = tosa.reshape %14, %2 : (tensor<48xi32>, !tosa.shape<4>) -> tensor<1x1x1x48xi32>
    %16 = tosa.add %11, %15 : (tensor<1x1x64x48xi32>, tensor<1x1x1x48xi32>) -> tensor<1x1x64x48xi32>
    %17 = tosa.rescale %16, %1, %0, %10, %9 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x1x64x48xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi16>) -> tensor<1x1x64x48xi16>
    %18 = tosa.clamp %17 {max_val = 32767 : i16, min_val = 0 : i16} : (tensor<1x1x64x48xi16>) -> tensor<1x1x64x48xi16>
    return %18 : tensor<1x1x64x48xi16>
  }
}
