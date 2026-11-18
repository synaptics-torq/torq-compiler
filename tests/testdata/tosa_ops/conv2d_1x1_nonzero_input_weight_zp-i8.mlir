// Regression: after ConvertNhwcOpToNchw, the weight-sum reduction input has
// dimension 1 where its indexing map and 16-element output require dimension 16.
// Isolated 1x1 pointwise convolution with nonzero input and weight zero points.
// Signed representation of uint8 activations (zp 0) and weights
// (zp 123); q_i8 = q_u8 - 128. Synthetic weights include -128 and 127, so
// folding the weight offset must preserve i16 headroom.
module {
  func.func @main(%arg0: tensor<1x8x8x32xi8>) -> tensor<1x8x8x16xi8> {
    %weights = "tosa.const"() <{values = dense<"0x802E58CC3FA7BB1820DF05EFE2E038EF2D30E12BB4C703AF4842ED26E7353CB801BDF9101343F00C2DB5293944C6E1C805EF34342B33D70B5A4E01BD3601FF3830CEE81CE925DACAE8573CB0BEDCD0FDC34AE41DB6E245C117FFEBEDC65AE8D9C1D93C32E1B34E4A0F47B1FE44AEE30133510250B0DA58371940AD26C9E3BFE73BE70C4BCBA921FEBD2DCF22F8C3F3EEEA3E46E4C6152F36D1A8F6231DB000D8B3A61946F6ACCFE658DE35204DA750ADE5BED8371849261B2DCE304AF9463813A6F64AFD08FC0C44021F57FE000F1B2A490234AD39AADBC9BAAED1D9DF272CEE051EC10DFB31D131D6050CEC1ED214A7B62814C935AE022FD8C5C3520EAF0A48B5C1F8B414B24F47A718F74728143F16042014BED7FE1C37102FA701CCE7D93306B6CD15161913C61313345123B344BE3A54E1EBB5D40DA957EE394B38DE02FD0954B5053C2BFFACACD5EA5A2AE9F94DFB14D4D8483915FCF130D105A64C2B45CDF014C4B057F5D81C1317E323E235F1A8D21A0242F82AA73F3CF0E0E74E48B7BCBEACF22ED1B8FCE64ACDD0DC1BE8DDBD1EBE1DEAC5D4E1B435CDDA0FC2DBE3FAD8E3C104DCB21635AE49F843562C4F23D819BAC3A9231A4CD8C0374A19CC4C2EE605C818B201194444E50DCDDC13574CD918EA37F5AD1A3CFBDE190EE739ECDDF2A95028DA0F38BD563234E235EC46DBEE4A29550B4D48A6D8F4DFDBD9597F"> : tensor<16x1x1x32xi8>}> : () -> tensor<16x1x1x32xi8>
    %bias = "tosa.const"() <{values = dense<[224, 371, -48, 94, -249, -548, 332, -259, 397, -10, -163, -528, 92, 243, 496, -401]> : tensor<16xi32>}> : () -> tensor<16xi32>
    %input_zp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %weight_zp = "tosa.const"() <{values = dense<-5> : tensor<1xi8>}> : () -> tensor<1xi8>
    %acc_zp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %output_zp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %multiplier = "tosa.const"() <{values = dense<1241689856> : tensor<1xi32>}> : () -> tensor<1xi32>
    %shift = "tosa.const"() <{values = dense<38> : tensor<1xi8>}> : () -> tensor<1xi8>
    %conv = tosa.conv2d %arg0, %weights, %bias, %input_zp, %weight_zp pad([0, 0, 0, 0]) stride([1, 1]) dilation([1, 1]) acc_type(i32) : (tensor<1x8x8x32xi8>, tensor<16x1x1x32xi8>, tensor<16xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x8x8x16xi32>
    %rescale = tosa.rescale %conv, %multiplier, %shift, %acc_zp, %output_zp scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<1x8x8x16xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x8x8x16xi8>
    %out = tosa.clamp %rescale min_val(-128 : i8) max_val(127 : i8) : (tensor<1x8x8x16xi8>) -> tensor<1x8x8x16xi8>
    return %out : tensor<1x8x8x16xi8>
  }
}
