// The multiply-with-a-table shape, but the shared activation is the function argument
// itself rather than the result of a leading rescale. Nothing produces the activation, so
// this case is what shows the fold takes the activation from wherever it comes from.
module {
  func.func @main(%arg0: tensor<1x8x8x16xi8>) -> tensor<1x8x8x16xi8> {
    %lut = "tosa.const"() <{values = dense<"0x8181818181818181828282828282828282828383838383838384848484848585858586868686878788888889898A8A8B8B8C8C8D8D8E8F8F909192939394959697989A9B9C9D9FA0A1A3A4A6A8A9ABADAFB1B3B5B7B9BBBEC0C2C5C7CACDCFD2D5D8DBDEE1E4E7EAEDF0F3F6FAFD0003060A0D101316191C1F2225282B2E313336393B3E40424547494B4D4F51535557585A5C5D5F60616364656668696A6B6C6D6D6E6F707171727373747475757676777778787879797A7A7A7A7B7B7B7B7C7C7C7C7C7D7D7D7D7D7D7D7E7E7E7E7E7E7E7E7E7E7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F"> : tensor<256xi8>}> : () -> tensor<256xi8>
    %out_mult = "tosa.const"() <{values = dense<1854308352> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_shift = "tosa.const"() <{values = dense<38> : tensor<1xi8>}> : () -> tensor<1xi8>
    %zero_i32 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %zero_i8 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %act_zp = "tosa.const"() <{values = dense<-18> : tensor<1xi8>}> : () -> tensor<1xi8>
    %tbl_zp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %out_zp = "tosa.const"() <{values = dense<-123> : tensor<1xi8>}> : () -> tensor<1xi8>
    %unit_mult = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %unit_shift = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>

    %sig = tosa.table %arg0, %lut : (tensor<1x8x8x16xi8>, tensor<256xi8>) -> tensor<1x8x8x16xi8>
    %direct = tosa.rescale %arg0, %unit_mult, %unit_shift, %act_zp, %zero_i32 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x8x8x16xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x8x8x16xi32>
    %scaled = tosa.rescale %sig, %unit_mult, %unit_shift, %tbl_zp, %zero_i32 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x8x8x16xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x8x8x16xi32>
    %prod = tosa.mul %direct, %scaled, %zero_i8 : (tensor<1x8x8x16xi32>, tensor<1x8x8x16xi32>, tensor<1xi8>) -> tensor<1x8x8x16xi32>
    %out = tosa.rescale %prod, %out_mult, %out_shift, %zero_i32, %out_zp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x8x8x16xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x8x8x16xi8>
    return %out : tensor<1x8x8x16xi8>
  }
}
