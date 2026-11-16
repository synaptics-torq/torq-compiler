// (a + b)^2 on two rescaled int8 inputs, as an i32 x i32 mul: the add form of an int8
// SQUARED_DIFFERENCE (Keras case squared_difference_mean_1x1x238 in keras_ops/add_mul_sub.py).
// The rescales, the add and the mul become i16 because |a| + |b| fits in i16.
module {
  func.func @main(%arg0: tensor<1x1x238xi8>, %arg1: tensor<1x1x1xi8>) -> tensor<1x1x238xi8> {
    %0 = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %1 = "tosa.const"() <{values = dense<49> : tensor<1xi8>}> : () -> tensor<1xi8>
    %2 = "tosa.const"() <{values = dense<1796499767> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %4 = "tosa.const"() <{values = dense<34> : tensor<1xi8>}> : () -> tensor<1xi8>
    %5 = "tosa.const"() <{values = dense<1122088820> : tensor<1xi32>}> : () -> tensor<1xi32>
    %6 = "tosa.const"() <{values = dense<31> : tensor<1xi8>}> : () -> tensor<1xi8>
    %7 = "tosa.const"() <{values = dense<18> : tensor<1xi8>}> : () -> tensor<1xi8>
    %8 = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %9 = "tosa.const"() <{values = dense<23> : tensor<1xi8>}> : () -> tensor<1xi8>
    %10 = "tosa.const"() <{values = dense<-15> : tensor<1xi8>}> : () -> tensor<1xi8>
    %11 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %12 = tosa.rescale %arg0, %8, %9, %10, %11 scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<1x1x238xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x1x238xi32>
    %13 = tosa.rescale %arg1, %8, %9, %7, %11 scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<1x1x1xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x1x1xi32>
    %14 = tosa.rescale %12, %8, %6, %11, %11 scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<1x1x238xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x1x238xi32>
    %15 = tosa.rescale %13, %5, %4, %11, %11 scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<1x1x1xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x1x1xi32>
    %16 = tosa.add %14, %15 : (tensor<1x1x238xi32>, tensor<1x1x1xi32>) -> tensor<1x1x238xi32>
    %17 = tosa.mul %16, %16, %3 : (tensor<1x1x238xi32>, tensor<1x1x238xi32>, tensor<1xi8>) -> tensor<1x1x238xi32>
    %18 = tosa.rescale %17, %2, %1, %11, %0 scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<1x1x238xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x1x238xi8>
    return %18 : tensor<1x1x238xi8>
  }
}
