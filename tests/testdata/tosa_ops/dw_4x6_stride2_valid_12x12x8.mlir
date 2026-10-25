// 4x6 stride-2 depthwise conv with no padding: the kernel borders differ per axis
module {
  func.func @main(%arg0: tensor<1x12x12x8xi8>) -> tensor<1x5x4x8xi8> {
    %w = "tosa.const"() <{values = dense<"0x594942831BCE059ED703975C89F59595A018AF7AC3564D526F388B74DF98BEDC76E595A6E0BED10331B37D77E4C682DE8B505162B54EC2E2D6970651083644BA93FD5FFA0DB32BD32949FCBBD09606632B8BC84C683FA378F8EC6BD91C10778F9DA1C47DC5AAED761976080AD4B72757BFD8F1749195A61BA6D76D5044A8307267DE8A97DAF64F8E41E43A637669AE2A0FC71C171DED5A72C2879B8DEA6CFA489E37A309EACF32889884CD7CDD972E9389413AA2BBA65AA7C40209019165B245"> : tensor<4x6x8x1xi8>}> : () -> tensor<4x6x8x1xi8>
    %b = "tosa.const"() <{values = dense<"0x2FFEFFFF9CFCFFFFF5FFFFFFAA020000F90000009DFCFFFF2C010000C8FEFFFF"> : tensor<8xi32>}> : () -> tensor<8xi32>
    %izp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %wzp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %m = "tosa.const"() <{values = dense<"0x3B4D7A4EB0038A5B0E02E4770BF6D37DA50B0749439BFB630E67BC70CB3C9150"> : tensor<8xi32>}> : () -> tensor<8xi32>
    %sh = "tosa.const"() <{values = dense<"0x2626262626262626"> : tensor<8xi8>}> : () -> tensor<8xi8>
    %rzp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %ozp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.depthwise_conv2d %arg0, %w, %b, %izp, %wzp {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x12x12x8xi8>, tensor<4x6x8x1xi8>, tensor<8xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x5x4x8xi32>
    %1 = tosa.rescale %0, %m, %sh, %rzp, %ozp {input_unsigned = false, output_unsigned = false, per_channel = true, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x5x4x8xi32>, tensor<8xi32>, tensor<8xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x5x4x8xi8>
    return %1 : tensor<1x5x4x8xi8>
  }
}
