// Int16 tensor minus a scalar, in the form that keeps the scalar a rescaled i16 constant up to the
// fused add. The add must fold the constant into its bias and not broadcast it.
func.func @main(%arg0: tensor<1x6x8x4xi16>) -> tensor<1x6x8x4xi16> {
  %m16 = "tosa.const"() <{values = dense<16384> : tensor<1xi16>}> : () -> tensor<1xi16>
  %s0 = "tosa.const"() <{values = dense<12> : tensor<1xi8>}> : () -> tensor<1xi8>
  %z16 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
  %z32 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  %c = "tosa.const"() <{values = dense<32767> : tensor<i16>}> : () -> tensor<i16>
  %shp = tosa.const_shape values(dense<1> : tensor<4xindex>) : () -> !tosa.shape<4>
  %mo = "tosa.const"() <{values = dense<16397> : tensor<1xi16>}> : () -> tensor<1xi16>
  %so = "tosa.const"() <{values = dense<28> : tensor<1xi8>}> : () -> tensor<1xi8>
  %x = tosa.rescale %arg0, %m16, %s0, %z16, %z32 scale32(false) rounding_mode<SINGLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<1x6x8x4xi16>, tensor<1xi16>, tensor<1xi8>, tensor<1xi16>, tensor<1xi32>) -> tensor<1x6x8x4xi32>
  %r = tosa.rescale %c, %m16, %s0, %z16, %z32 scale32(false) rounding_mode<SINGLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<i16>, tensor<1xi16>, tensor<1xi8>, tensor<1xi16>, tensor<1xi32>) -> tensor<i32>
  %b = tosa.reshape %r, %shp : (tensor<i32>, !tosa.shape<4>) -> tensor<1x1x1x1xi32>
  %o = tosa.sub %x, %b : (tensor<1x6x8x4xi32>, tensor<1x1x1x1xi32>) -> tensor<1x6x8x4xi32>
  %y = tosa.rescale %o, %mo, %so, %z32, %z16 scale32(false) rounding_mode<SINGLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<1x6x8x4xi32>, tensor<1xi16>, tensor<1xi8>, tensor<1xi32>, tensor<1xi16>) -> tensor<1x6x8x4xi16>
  return %y : tensor<1x6x8x4xi16>
}
