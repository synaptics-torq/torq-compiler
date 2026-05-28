// Per-channel bias add where the bias constant has a chained rescale that
// flows through tosa.reshape before broadcasting on a non-unit dimension.
// Exercises peeling through the broadcast-only linalg.generic wrapper that
// the broadcast-elementwise pass materializes, and re-broadcasting the
// underlying constant back to the add output shape after the deeper peel.
// Extracted from NNNR3_0079_0.0960_int8x8 TFLite model (layer ADD_605).
module {
  func.func @main(%arg0: tensor<1x1x32x48xi8>) -> tensor<1x1x32x48xi8> {
    %0 = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %1 = "tosa.const"() <{values = dense<48> : tensor<1xi8>}> : () -> tensor<1xi8>
    %2 = "tosa.const"() <{values = dense<1193829374> : tensor<1xi32>}> : () -> tensor<1xi32>
    %3 = tosa.const_shape  {values = dense<[1, 1, 1, 48]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %4 = "tosa.const"() <{values = dense<35> : tensor<1xi8>}> : () -> tensor<1xi8>
    %5 = "tosa.const"() <{values = dense<1122577871> : tensor<1xi32>}> : () -> tensor<1xi32>
    %6 = "tosa.const"() <{values = dense<-16> : tensor<1xi8>}> : () -> tensor<1xi8>
    %7 = "tosa.const"() <{values = dense<10> : tensor<1xi8>}> : () -> tensor<1xi8>
    %8 = "tosa.const"() <{values = dense<[-112, -20, -128, 84, 28, -87, 127, 9, 21, 56, 7, -4, 1, -43, 31, -19, 23, 97, -4, -21, -37, -13, 16, -11, 19, -2, 58, -43, -6, -36, -30, -4, 25, -48, -69, 27, 91, 61, 20, -26, 26, 17, -108, 2, 10, 3, -42, -20]> : tensor<48xi8>}> : () -> tensor<48xi8>
    %9 = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %10 = "tosa.const"() <{values = dense<11> : tensor<1xi8>}> : () -> tensor<1xi8>
    %11 = "tosa.const"() <{values = dense<13> : tensor<1xi8>}> : () -> tensor<1xi8>
    %12 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %13 = tosa.rescale %arg0, %9, %10, %11, %12 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x1x32x48xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x1x32x48xi32>
    %14 = tosa.rescale %8, %9, %7, %6, %12 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<48xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<48xi32>
    %15 = tosa.rescale %14, %5, %4, %12, %12 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<48xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<48xi32>
    %16 = tosa.reshape %15, %3 : (tensor<48xi32>, !tosa.shape<4>) -> tensor<1x1x1x48xi32>
    %17 = tosa.add %13, %16 : (tensor<1x1x32x48xi32>, tensor<1x1x1x48xi32>) -> tensor<1x1x32x48xi32>
    %18 = tosa.rescale %17, %2, %1, %12, %0 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x1x32x48xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x1x32x48xi8>
    %19 = tosa.clamp %18 {max_val = 127 : i8, min_val = -128 : i8} : (tensor<1x1x32x48xi8>) -> tensor<1x1x32x48xi8>
    return %19 : tensor<1x1x32x48xi8>
  }
}
