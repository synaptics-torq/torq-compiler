// The chained-table shape a quantized mish legalizes to: a table, a rescale, an add of a
// broadcast constant, a rescale back to i8, a second table, and a final rescale. Nothing
// here is a diamond and the two tables are three ops apart, so this case is what shows the
// fold walks the chain rather than matching a fixed shape.
module {
  func.func @main(%arg0: tensor<1x8x8x16xi8>) -> tensor<1x8x8x16xi8> {
    %lut1 = "tosa.const"() <{values = dense<"0xD8D8D8D8D8D8D8D8D8D8D8D8D8D8D8D8D8D8D8D8D8D8D8D8D8D8D8D8D8D8D8D8D8D8D8D8D8D9D9D9D9D9D9D9D9D9D9D9D9D9D9D9D9D9D9D9D9D9D9DADADADADADADADADADADADBDBDBDBDBDBDBDCDCDCDCDCDCDDDDDDDDDEDEDEDEDFDFDFDFE0E0E0E1E1E1E2E2E3E3E4E4E4E5E5E6E6E7E7E8E9E9EAEAEBEBECEDEDEEEFEFF0F1F2F2F3F4F5F5F6F7F8F8F9FAFBFCFDFDFEFF00010203030405060708090A0B0C0D0D0E0F101112131415161718191A1B1C1D1E1F20212222232425262728292A2B2C2D2E2F303132333435363738393A3B3C3D3E3F404142434445464748494A4B4C4D4E4F505152535455565758595A5B5C5D5E5F60616263646566676869"> : tensor<256xi8>}> : () -> tensor<256xi8>
    %lut2 = "tosa.const"() <{values = dense<"0xC9C9C9CACACBCBCBCCCCCCCDCDCDCECECECFCFCFD0D0D1D1D1D2D2D2D3D3D4D4D4D5D5D6D6D6D7D7D8D8D8D9D9DADADADBDBDCDCDDDDDDDEDEDFDFE0E0E0E1E1E2E2E3E3E4E4E5E5E5E6E6E7E7E8E8E9E9EAEAEBEBEBECECEDEDEEEEEFEFF0F0F1F1F2F2F3F3F4F4F5F5F6F6F7F7F8F8F9F9FAFAFBFBFCFCFDFDFEFEFFFF0000010102020303030404050506060707080809090A0A0B0B0C0C0D0D0E0E0F0F10101111121213131414151516161717181819191A1A1B1B1B1C1C1D1D1E1E1F1F20202121212222232324242525262626272728282929292A2A2B2B2C2C2C2D2D2E2E2E2F2F303030313132323233333434343535353636373737383838393939"> : tensor<256xi8>}> : () -> tensor<256xi8>
    %out_mult = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_shift = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>
    %zero_i32 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %act_zp = "tosa.const"() <{values = dense<-18> : tensor<1xi8>}> : () -> tensor<1xi8>
    %mid_zp = "tosa.const"() <{values = dense<5> : tensor<1xi8>}> : () -> tensor<1xi8>
    %out_zp = "tosa.const"() <{values = dense<-7> : tensor<1xi8>}> : () -> tensor<1xi8>
    %zero_i8 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %bias = "tosa.const"() <{values = dense<64> : tensor<1x1x1x1xi32>}> : () -> tensor<1x1x1x1xi32>
    %unit_mult = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %unit_shift = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>

    %act = tosa.rescale %arg0, %unit_mult, %unit_shift, %zero_i8, %act_zp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x8x8x16xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x8x8x16xi8>
    %t1 = tosa.table %act, %lut1 : (tensor<1x8x8x16xi8>, tensor<256xi8>) -> tensor<1x8x8x16xi8>
    %widened = tosa.rescale %t1, %unit_mult, %unit_shift, %act_zp, %zero_i32 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x8x8x16xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x8x8x16xi32>
    // The other addend is a constant, so the sum still depends on the activation only.
    %shifted = tosa.add %widened, %bias : (tensor<1x8x8x16xi32>, tensor<1x1x1x1xi32>) -> tensor<1x8x8x16xi32>
    %narrowed = tosa.rescale %shifted, %unit_mult, %unit_shift, %zero_i32, %mid_zp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x8x8x16xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x8x8x16xi8>
    %t2 = tosa.table %narrowed, %lut2 : (tensor<1x8x8x16xi8>, tensor<256xi8>) -> tensor<1x8x8x16xi8>
    %out = tosa.rescale %t2, %out_mult, %out_shift, %mid_zp, %out_zp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x8x8x16xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x8x8x16xi8>
    return %out : tensor<1x8x8x16xi8>
  }
}
