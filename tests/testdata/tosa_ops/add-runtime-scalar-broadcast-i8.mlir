// Quantized add of a tensor and a runtime 1-element tensor that broadcasts on every dim.
// The table makes the scalar a runtime value of type tensor<1xi8>, which reaches the add
// dispatch as a rank-1 argument. The fused add peels the broadcast down to that scalar.
// It must broadcast it again to the output shape, through a collapse to rank 0. Before the
// fix the add kept the 1-element operand and failed with "Add input shapes must match".
// Extracted from the COFI_LITE_221k int8x8 TFLite model (full model).
module {
  func.func @main(%arg0: tensor<1x1x238xi8>, %arg1: tensor<1xi8>) -> tensor<1x1x238xi8> {
    %0 = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %1 = "tosa.const"() <{values = dense<11> : tensor<1xi8>}> : () -> tensor<1xi8>
    %2 = "tosa.const"() <{values = dense<-22> : tensor<1xi8>}> : () -> tensor<1xi8>
    %3 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "tosa.const"() <{values = dense<10> : tensor<1xi8>}> : () -> tensor<1xi8>
    %5 = "tosa.const"() <{values = dense<110> : tensor<1xi8>}> : () -> tensor<1xi8>
    %6 = "tosa.const"() <{values = dense<1875082765> : tensor<1xi32>}> : () -> tensor<1xi32>
    %7 = "tosa.const"() <{values = dense<36> : tensor<1xi8>}> : () -> tensor<1xi8>
    %8 = "tosa.const"() <{values = dense<1090068368> : tensor<1xi32>}> : () -> tensor<1xi32>
    %9 = "tosa.const"() <{values = dense<49> : tensor<1xi8>}> : () -> tensor<1xi8>
    %10 = "tosa.const"() <{values = dense<-15> : tensor<1xi8>}> : () -> tensor<1xi8>
    %11 = tosa.const_shape values(dense<1> : tensor<3xindex>) : () -> !tosa.shape<3>
    %t = "tosa.const"() <{values = dense<"0x7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7C7875726F6C696664615F5C5A585653514F4E4C4A4846454341403E3D3B3A39373635333231302E2D2C2B2A292827262524232221201F1E1D1D1C1B1A19191817161615141313121111100F0F0E0D0D0C0C0B0A0A0909080807070605050404030302020201010000FFFFFEFEFDFDFDFCFCFBFBFBFAFAF9F9F9F8F8F7F7F7F6F6F6F5F5F5F4F4F4F3F3F2F2F2F2F1F1F1F0F0F0EFEFEFEEEEEEEEEDEDEDECECECECEBEBEBEBEAEAEAE9E9E9E9E8E8E8E8E7E7E7E7E7E6E6E6E6E5E5E5E5E4E4E4E4E4E3E3E3E3E3E2E2E2E2E1E1E1E1E1E0E0E0E0E0E0DFDFDFDFDF"> : tensor<256xi8>}> : () -> tensor<256xi8>
    %s = tosa.table %arg1, %t : (tensor<1xi8>, tensor<256xi8>) -> tensor<1xi8>
    %12 = tosa.reshape %s, %11 : (tensor<1xi8>, !tosa.shape<3>) -> tensor<1x1x1xi8>
    %13 = tosa.rescale %arg0, %0, %1, %2, %3 scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<1x1x238xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x1x238xi32>
    %14 = tosa.rescale %12, %0, %4, %5, %3 scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<1x1x1xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x1x1xi32>
    %15 = tosa.rescale %14, %6, %7, %3, %3 scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<1x1x1xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x1x1xi32>
    %16 = tosa.add %13, %15 : (tensor<1x1x238xi32>, tensor<1x1x1xi32>) -> tensor<1x1x238xi32>
    %17 = tosa.rescale %16, %8, %9, %3, %10 scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<1x1x238xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x1x238xi8>
    return %17 : tensor<1x1x238xi8>
  }
}
