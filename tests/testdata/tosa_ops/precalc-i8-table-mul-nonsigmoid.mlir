// The multiply-with-a-table shape of precalc-i8-table-mul.mlir over a sawtooth table
// instead of a logistic one. The fold matches on the shape of the chain and reads the
// table as data, so a table that is neither monotone nor smooth has to fold to the same
// values the llvm-cpu reference computes at runtime.
module {
  func.func @main(%arg0: tensor<1x8x8x16xi8>) -> tensor<1x8x8x16xi8> {
    %lut = "tosa.const"() <{values = dense<"0xFE03080D12171C21262B30353A3F44494E53585D62676C71767B81868B90959A9FA4A9AEB3B8BDC2C7CCD1D6DBE0E5EAEFF4F9FE03080D12171C21262B30353A3F44494E53585D62676C71767B81868B90959A9FA4A9AEB3B8BDC2C7CCD1D6DBE0E5EAEFF4F9FE03080D12171C21262B30353A3F44494E53585D62676C71767B81868B90959A9FA4A9AEB3B8BDC2C7CCD1D6DBE0E5EAEFF4F9FE03080D12171C21262B30353A3F44494E53585D62676C71767B81868B90959A9FA4A9AEB3B8BDC2C7CCD1D6DBE0E5EAEFF4F9FE03080D12171C21262B30353A3F44494E53585D62676C71767B81868B90959A9FA4A9AEB3B8BDC2C7CCD1D6DBE0E5EAEFF4F9FE"> : tensor<256xi8>}> : () -> tensor<256xi8>
    %out_mult = "tosa.const"() <{values = dense<1854308352> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_shift = "tosa.const"() <{values = dense<38> : tensor<1xi8>}> : () -> tensor<1xi8>
    %zero_i32 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %zero_i8 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %act_zp = "tosa.const"() <{values = dense<-18> : tensor<1xi8>}> : () -> tensor<1xi8>
    %tbl_zp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %out_zp = "tosa.const"() <{values = dense<-123> : tensor<1xi8>}> : () -> tensor<1xi8>
    %unit_mult = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %unit_shift = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>

    %act = tosa.rescale %arg0, %unit_mult, %unit_shift, %zero_i8, %act_zp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x8x8x16xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x8x8x16xi8>
    %tbl = tosa.table %act, %lut : (tensor<1x8x8x16xi8>, tensor<256xi8>) -> tensor<1x8x8x16xi8>
    %direct = tosa.rescale %act, %unit_mult, %unit_shift, %act_zp, %zero_i32 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x8x8x16xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x8x8x16xi32>
    %scaled = tosa.rescale %tbl, %unit_mult, %unit_shift, %tbl_zp, %zero_i32 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x8x8x16xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x8x8x16xi32>
    %prod = tosa.mul %direct, %scaled, %zero_i8 : (tensor<1x8x8x16xi32>, tensor<1x8x8x16xi32>, tensor<1xi8>) -> tensor<1x8x8x16xi32>
    %out = tosa.rescale %prod, %out_mult, %out_shift, %zero_i32, %out_zp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x8x8x16xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x8x8x16xi8>
    return %out : tensor<1x8x8x16xi8>
  }
}
