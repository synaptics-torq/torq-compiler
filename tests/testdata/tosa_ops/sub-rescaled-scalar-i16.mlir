// Int16 scalar minus tensor (1 - sigmoid(x)). Only the fused lowering handles a scalar
// operand; routing it to the unfused one leaves a 1x1 operand against a 1x32 output and
// fails to legalize. Guards the scalar exclusion in the broadcast reroute.
// Extracted from NNNR3_0079_0.0960_int16x8 TFLite model (layer SUB_54).
module {
  func.func @main(%arg0: tensor<1x32xi16>) -> tensor<1x32xi16> {
    %0 = "tosa.const"() <{values = dense<44> : tensor<1xi8>}> : () -> tensor<1xi8>
    %1 = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
    %2 = "tosa.const"() <{values = dense<32> : tensor<1xi8>}> : () -> tensor<1xi8>
    %3 = "tosa.const"() <{values = dense<2147418114> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "tosa.const"() <{values = dense<15> : tensor<1x1xi32>}> : () -> tensor<1x1xi32>
    %5 = "tosa.const"() <{values = dense<32767> : tensor<i16>}> : () -> tensor<i16>
    %6 = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %7 = "tosa.const"() <{values = dense<16> : tensor<1xi8>}> : () -> tensor<1xi8>
    %8 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
    %9 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %10 = tosa.rescale %5, %6, %7, %8, %9 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<i16>, tensor<1xi32>, tensor<1xi8>, tensor<1xi16>, tensor<1xi32>) -> tensor<i32>
    %11 = tosa.cast %arg0 : (tensor<1x32xi16>) -> tensor<1x32xi32>
    %12 = tosa.logical_left_shift %11, %4 : (tensor<1x32xi32>, tensor<1x1xi32>) -> tensor<1x32xi32>
    %13 = tosa.rescale %12, %3, %2, %9, %9 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x32xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x32xi32>
    %14 = tosa.reshape %10, %1 : (tensor<i32>, !tosa.shape<2>) -> tensor<1x1xi32>
    %15 = tosa.sub %14, %13 : (tensor<1x1xi32>, tensor<1x32xi32>) -> tensor<1x32xi32>
    %16 = tosa.rescale %15, %6, %0, %9, %8 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x32xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi16>) -> tensor<1x32xi16>
    return %16 : tensor<1x32xi16>
  }
}
