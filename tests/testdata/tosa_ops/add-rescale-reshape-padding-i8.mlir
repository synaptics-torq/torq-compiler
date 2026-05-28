// Rank-padding via tosa.reshape between a double-rescale chain and tosa.add.
// The right input has rank 2 (16x16) and is reshaped to rank 3 (1x16x16) to
// match the left input. Exercises peeling through tensor.expand_shape so the
// rescale chain stays foldable on both inputs of the add.
// Extracted from NNNR3_0079_0.0960_int8x8 TFLite model (layer ADD_539).
module {
  func.func @main(%arg0: tensor<1x16x16xi8>, %arg1: tensor<16x16xi8>) -> tensor<1x16x16xi8> {
    %0 = "tosa.const"() <{values = dense<-14> : tensor<1xi8>}> : () -> tensor<1xi8>
    %1 = "tosa.const"() <{values = dense<50> : tensor<1xi8>}> : () -> tensor<1xi8>
    %2 = "tosa.const"() <{values = dense<2011511696> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = tosa.const_shape  {values = dense<[1, 16, 16]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %4 = "tosa.const"() <{values = dense<32> : tensor<1xi8>}> : () -> tensor<1xi8>
    %5 = "tosa.const"() <{values = dense<1423433540> : tensor<1xi32>}> : () -> tensor<1xi32>
    %6 = "tosa.const"() <{values = dense<-30> : tensor<1xi8>}> : () -> tensor<1xi8>
    %7 = "tosa.const"() <{values = dense<10> : tensor<1xi8>}> : () -> tensor<1xi8>
    %8 = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %9 = "tosa.const"() <{values = dense<11> : tensor<1xi8>}> : () -> tensor<1xi8>
    %10 = "tosa.const"() <{values = dense<-19> : tensor<1xi8>}> : () -> tensor<1xi8>
    %11 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %12 = tosa.rescale %arg0, %8, %9, %10, %11 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x16x16xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x16x16xi32>
    %13 = tosa.rescale %arg1, %8, %7, %6, %11 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<16x16xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<16x16xi32>
    %14 = tosa.rescale %13, %5, %4, %11, %11 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<16x16xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<16x16xi32>
    %15 = tosa.reshape %14, %3 : (tensor<16x16xi32>, !tosa.shape<3>) -> tensor<1x16x16xi32>
    %16 = tosa.add %12, %15 : (tensor<1x16x16xi32>, tensor<1x16x16xi32>) -> tensor<1x16x16xi32>
    %17 = tosa.rescale %16, %2, %1, %11, %0 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x16x16xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x16x16xi8>
    return %17 : tensor<1x16x16xi8>
  }
}
