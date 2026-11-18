// Stride-2 3x3 transpose convolution with a nonzero input zero point only.
// Its weight sum reads the convolution filter and must stay a compile-time
// constant on NSS. Synthetic weights include both i8 endpoints.
module {
  func.func @main(%arg0: tensor<1x8x8x8xi8>) -> tensor<1x17x17x4xi8> {
    %weights = "tosa.const"() <{values = dense<"0xECDE20F15BADDDFEC8BB0FA948E1409049F859324D269B79E4620222AE654AF1EB4CAAC8A406DF7A64194301EC49DCCFA4544A8457BC63ADA8C16F7229861ECC3A136033D046BDE2BCB8E75AF19F40900FBB67FEE971F53AEF10FF05EF31F92039D6AB9719B3DB48F622A54C188BF866F8E1876DB918ED0C9E51A920C12C98A8671A50DB439337A224FA21C39A43F69FEB83617FE6D0210241B5E1B572E6244A1E897C56CE7D640FEEDCC1BB9FAE7AD2B65A1880C049C62A1484A091523CA12CA8D02F5CCCB585ED119D2925A896B2ADAE44B9004629EB3B851BD9B2AD32769DCDC5FDF22C9D019B4D26613D2FDE58E3BEB50305C014B46F888AD468B67DD6D89B395D53A526F3C9431F5C605B9B1EFAAC85B4582F47668BE83E3C31A3A0ED06"> : tensor<4x3x3x8xi8>}> : () -> tensor<4x3x3x8xi8>
    %bias = "tosa.const"() <{values = dense<[-183, 189, -278, -165]> : tensor<4xi32>}> : () -> tensor<4xi32>
    %input_zp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %weight_zp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %conv = tosa.transpose_conv2d %arg0, %weights, %bias, %input_zp, %weight_zp out_pad([0, 0, 0, 0]) stride([2, 2]) acc_type(i32) : (tensor<1x8x8x8xi8>, tensor<4x3x3x8xi8>, tensor<4xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x17x17x4xi32>
    %acc_zp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %output_zp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %multiplier = "tosa.const"() <{values = dense<1241689856> : tensor<1xi32>}> : () -> tensor<1xi32>
    %shift = "tosa.const"() <{values = dense<41> : tensor<1xi8>}> : () -> tensor<1xi8>
    %rescale = tosa.rescale %conv, %multiplier, %shift, %acc_zp, %output_zp scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(false) input_unsigned(false) output_unsigned(false) : (tensor<1x17x17x4xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x17x17x4xi8>
    return %rescale : tensor<1x17x17x4xi8>
  }
}
