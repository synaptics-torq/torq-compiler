// Regression: with a nonzero depthwise weight zero point, the correction that
// sums the input under each window is not folded and fails NSS-only compilation.
// Isolated 3x3 depthwise convolution with nonzero input and weight zero points.
// Signed representation of uint8 activations (zp 0) and weights
// (zp 124); q_i8 = q_u8 - 128. Synthetic weights include -128 and 127, so
// folding the weight offset needs i16 headroom.
module {
  func.func @main(%arg0: tensor<1x64x32x32xi8>) -> tensor<1x64x32x32xi8> {
    %weights = "tosa.const"() <{values = dense<"0x80E101D503505817AEBBEB1D43A6D5071DCF34D44AAC384B2414A853DE5436FACEF7AC3D09AC5755AD4DCCCA16B1E81BEEA8ED29A9131CCC4F294B3F0603D13F0A24DF3DAA24DAE53813A8D1AC23F356BDC13E23CCEB25DBBC3BF02FC8F016DB0FC6D8D012DC2B13B60753D522B1F1A6CA34DD2AD3421819F1FBC0DAF617C22BEBBEC7C0CA22D9BCC45922C515AC2213300F495AEF05C5A6C1E42E5549E1C2FAF8EAB22BBEA9E45800DC23C711350AC7D00A37B9ADDD1C34CBE722004B3534C0203BE5CDBFF233F9D53738D40AC214D40D394AB2D7BC01FE470BBDB7F80F3047D100EE080C003EF3D43DA810BEDBB5FCA7523EB8F7A830EA2639F2BF2A33DD5021B8CF44D353ECACC1D8C5DD16EEE6280D452A4437BCDFC12608B654B9D7E27F"> : tensor<3x3x32x1xi8>}> : () -> tensor<3x3x32x1xi8>
    %bias = "tosa.const"() <{values = dense<[168, 346, 344, 291, -296, -561, -376, -293, -268, -457, -251, 260, -12, -467, -115, -165, 277, -351, -586, 471, 292, 359, -26, 126, 476, -338, 441, 161, 589, -12, 375, 51]> : tensor<32xi32>}> : () -> tensor<32xi32>
    %input_zp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %weight_zp = "tosa.const"() <{values = dense<-4> : tensor<1xi8>}> : () -> tensor<1xi8>
    %acc_zp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %output_zp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %multiplier = "tosa.const"() <{values = dense<1314186112> : tensor<1xi32>}> : () -> tensor<1xi32>
    %shift = "tosa.const"() <{values = dense<38> : tensor<1xi8>}> : () -> tensor<1xi8>
    %conv = tosa.depthwise_conv2d %arg0, %weights, %bias, %input_zp, %weight_zp pad([1, 1, 1, 1]) stride([1, 1]) dilation([1, 1]) acc_type(i32) : (tensor<1x64x32x32xi8>, tensor<3x3x32x1xi8>, tensor<32xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x64x32x32xi32>
    %rescale = tosa.rescale %conv, %multiplier, %shift, %acc_zp, %output_zp scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<1x64x32x32xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x64x32x32xi8>
    %out = tosa.clamp %rescale min_val(-128 : i8) max_val(127 : i8) : (tensor<1x64x32x32xi8>) -> tensor<1x64x32x32xi8>
    return %out : tensor<1x64x32x32xi8>
  }
}
