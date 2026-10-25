// 2x3 stride-2 depthwise conv with padding 0 then 1 along H and same along W: the kernel borders differ per axis
module {
  func.func @main(%arg0: tensor<1x11x11x8xi8>) -> tensor<1x6x6x8xi8> {
    %w = "tosa.const"() <{values = dense<"0xFF684BC527F55DBDDB1028B56537DB62AE1276B7B01351B9F0899DA0B450619F8437C946E38B4FC984CECE4CC2A98C15"> : tensor<2x3x8x1xi8>}> : () -> tensor<2x3x8x1xi8>
    %b = "tosa.const"() <{values = dense<"0xCC0200006C000000100200006B02000092FCFFFF78000000A5FFFFFF58FEFFFF"> : tensor<8xi32>}> : () -> tensor<8xi32>
    %izp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %wzp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %m = "tosa.const"() <{values = dense<"0x92D4F45CC7E46C5AB8137A5F0E185C7451E6D4748EAE18681F5B756D3087617D"> : tensor<8xi32>}> : () -> tensor<8xi32>
    %sh = "tosa.const"() <{values = dense<"0x2626262626262626"> : tensor<8xi8>}> : () -> tensor<8xi8>
    %rzp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %ozp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.depthwise_conv2d %arg0, %w, %b, %izp, %wzp {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 1, 1>, stride = array<i64: 2, 2>} : (tensor<1x11x11x8xi8>, tensor<2x3x8x1xi8>, tensor<8xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x6x6x8xi32>
    %1 = tosa.rescale %0, %m, %sh, %rzp, %ozp {input_unsigned = false, output_unsigned = false, per_channel = true, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x6x6x8xi32>, tensor<8xi32>, tensor<8xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x6x6x8xi8>
    return %1 : tensor<1x6x6x8xi8>
  }
}
