// A quantized SiLU (x * sigmoid(x)) taken from YOLOv8n-MIXED: one i8 activation feeds
// both a sigmoid table and an identity rescale, the two arms are multiplied, and the
// product is rescaled back to i8.
//
// Both ends being i8 is what lets PreCalcI8TablePattern replay the whole chain at compile
// time and store the result in one 256-entry table. Comparing against the llvm-cpu
// reference, which keeps the chain at runtime, is what checks that the precomputation is
// exact.
//
// Use --torq-disable-precalc-i8-table to keep the chain and compare the two forms.
module {
  func.func @main(%arg0: tensor<1x8x8x16xi8>) -> tensor<1x8x8x16xi8> {
    %lut = "tosa.const"() <{values = dense<"0x80808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808080808081818181818181818181818282828282838383848484858586868788898A8B8C8D8E9091939597999B9EA0A3A7AAAEB2B6BABFC4C9CFD4DAE0E6EDF3FA00060D131A20262C31373C41464A4E5256595D60626567696B6D6F7072737475767778797A7A7B7B7C7C7C7D7D7D7E7E7E7E7E7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F7F"> : tensor<256xi8>}> : () -> tensor<256xi8>
    %out_mult = "tosa.const"() <{values = dense<1854308352> : tensor<1xi32>}> : () -> tensor<1xi32>
    %out_shift = "tosa.const"() <{values = dense<38> : tensor<1xi8>}> : () -> tensor<1xi8>
    %zero_i32 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %zero_i8 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %act_zp = "tosa.const"() <{values = dense<-18> : tensor<1xi8>}> : () -> tensor<1xi8>
    %tbl_zp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %out_zp = "tosa.const"() <{values = dense<-123> : tensor<1xi8>}> : () -> tensor<1xi8>
    %unit_mult = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %unit_shift = "tosa.const"() <{values = dense<30> : tensor<1xi8>}> : () -> tensor<1xi8>

    // The activation both arms share.
    %act = tosa.rescale %arg0, %unit_mult, %unit_shift, %zero_i8, %act_zp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x8x8x16xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x8x8x16xi8>
    %sig = tosa.table %act, %lut : (tensor<1x8x8x16xi8>, tensor<256xi8>) -> tensor<1x8x8x16xi8>
    %direct = tosa.rescale %act, %unit_mult, %unit_shift, %act_zp, %zero_i32 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x8x8x16xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x8x8x16xi32>
    %scaled = tosa.rescale %sig, %unit_mult, %unit_shift, %tbl_zp, %zero_i32 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x8x8x16xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x8x8x16xi32>
    %prod = tosa.mul %direct, %scaled, %zero_i8 : (tensor<1x8x8x16xi32>, tensor<1x8x8x16xi32>, tensor<1xi8>) -> tensor<1x8x8x16xi32>
    %out = tosa.rescale %prod, %out_mult, %out_shift, %zero_i32, %out_zp {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x8x8x16xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x8x8x16xi8>
    return %out : tensor<1x8x8x16xi8>
  }
}
