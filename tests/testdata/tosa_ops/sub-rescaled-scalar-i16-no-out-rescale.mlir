// Int16 tensor minus a rescaled int16 scalar, with an i32 result and no rescale after the sub. The
// fused lowering folds the rescales of the scalar into its weights only when a rescale follows the
// sub, so the broadcast pass computes the scalar at compile time for the unfused one; left at
// 1x1x1x1, it fails to legalize against the 1x6x8x4 output
// (synaptics-torq/torq-compiler-dev#2869).
func.func @main(%arg0: tensor<1x6x8x4xi16>) -> tensor<1x6x8x4xi32> {
  %m16 = "tosa.const"() <{values = dense<16384> : tensor<1xi16>}> : () -> tensor<1xi16>
  %s0 = "tosa.const"() <{values = dense<12> : tensor<1xi8>}> : () -> tensor<1xi8>
  %z16 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
  %z32 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  %c = "tosa.const"() <{values = dense<32767> : tensor<i16>}> : () -> tensor<i16>
  %shp = tosa.const_shape {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
  %x = tosa.rescale %arg0, %m16, %s0, %z16, %z32 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = SINGLE_ROUND, scale32 = false} : (tensor<1x6x8x4xi16>, tensor<1xi16>, tensor<1xi8>, tensor<1xi16>, tensor<1xi32>) -> tensor<1x6x8x4xi32>
  %r = tosa.rescale %c, %m16, %s0, %z16, %z32 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = SINGLE_ROUND, scale32 = false} : (tensor<i16>, tensor<1xi16>, tensor<1xi8>, tensor<1xi16>, tensor<1xi32>) -> tensor<i32>
  %b = tosa.reshape %r, %shp : (tensor<i32>, !tosa.shape<4>) -> tensor<1x1x1x1xi32>
  %o = tosa.sub %x, %b : (tensor<1x6x8x4xi32>, tensor<1x1x1x1xi32>) -> tensor<1x6x8x4xi32>
  return %o : tensor<1x6x8x4xi32>
}
