// Quantized add of a tensor and a runtime 1x1x1 tensor. Every output dim is larger than 1, so
// the fused add keeps no dim of the 1x1x1 operand. It collapses the operand to rank 0 and
// broadcasts it again to the output shape. Before the fix the add failed with "Add input shapes
// must match". See add-runtime-scalar-broadcast-i8.mlir.
module {
  func.func @main(%arg0: tensor<2x4x64xi8>, %arg1: tensor<1x1x1xi8>) -> tensor<2x4x64xi8> {
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
    %13 = tosa.rescale %arg0, %0, %1, %2, %3 scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<2x4x64xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<2x4x64xi32>
    %14 = tosa.rescale %arg1, %0, %4, %5, %3 scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<1x1x1xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x1x1xi32>
    %15 = tosa.rescale %14, %6, %7, %3, %3 scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<1x1x1xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x1x1xi32>
    %16 = tosa.add %13, %15 : (tensor<2x4x64xi32>, tensor<1x1x1xi32>) -> tensor<2x4x64xi32>
    %17 = tosa.rescale %16, %8, %9, %3, %10 scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<2x4x64xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<2x4x64xi8>
    return %17 : tensor<2x4x64xi8>
  }
}
