// Int16 tensor minus scalar, as the TFLite converter lowers it when the scalar is not at half
// the scale of the sum: cast to i32, shift left by 15, rescale. The cast folds into an i32
// constant, which the fused lowering rejects, so the broadcast pass computes the scalar at
// compile time for the unfused one; left at 1x1x1x1, it fails to legalize against the
// 1x6x8x4 output.
func.func @main(%arg0: tensor<1x6x8x4xi16>) -> tensor<1x6x8x4xi16> {
  %m16 = "tosa.const"() <{values = dense<16384> : tensor<1xi16>}> : () -> tensor<1xi16>
  %s0 = "tosa.const"() <{values = dense<12> : tensor<1xi8>}> : () -> tensor<1xi8>
  %z16 = "tosa.const"() <{values = dense<0> : tensor<1xi16>}> : () -> tensor<1xi16>
  %z32 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
  %mr = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
  %sr = "tosa.const"() <{values = dense<33> : tensor<1xi8>}> : () -> tensor<1xi8>
  %c = "tosa.const"() <{values = dense<32767> : tensor<i16>}> : () -> tensor<i16>
  %sh = "tosa.const"() <{values = dense<15> : tensor<i32>}> : () -> tensor<i32>
  %shp = tosa.const_shape values(dense<1> : tensor<4xindex>) : () -> !tosa.shape<4>
  %mo = "tosa.const"() <{values = dense<16397> : tensor<1xi16>}> : () -> tensor<1xi16>
  %so = "tosa.const"() <{values = dense<28> : tensor<1xi8>}> : () -> tensor<1xi8>
  %x = tosa.rescale %arg0, %m16, %s0, %z16, %z32 scale32(false) rounding_mode<SINGLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<1x6x8x4xi16>, tensor<1xi16>, tensor<1xi8>, tensor<1xi16>, tensor<1xi32>) -> tensor<1x6x8x4xi32>
  %cc = tosa.cast %c : (tensor<i16>) -> tensor<i32>
  %l = tosa.logical_left_shift %cc, %sh : (tensor<i32>, tensor<i32>) -> tensor<i32>
  %r = tosa.rescale %l, %mr, %sr, %z32, %z32 scale32(true) rounding_mode<SINGLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<i32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<i32>
  %b = tosa.reshape %r, %shp : (tensor<i32>, !tosa.shape<4>) -> tensor<1x1x1x1xi32>
  %o = tosa.sub %x, %b : (tensor<1x6x8x4xi32>, tensor<1x1x1x1xi32>) -> tensor<1x6x8x4xi32>
  %y = tosa.rescale %o, %mo, %so, %z32, %z16 scale32(false) rounding_mode<SINGLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<1x6x8x4xi32>, tensor<1xi16>, tensor<1xi8>, tensor<1xi32>, tensor<1xi16>) -> tensor<1x6x8x4xi16>
  return %y : tensor<1x6x8x4xi16>
}
