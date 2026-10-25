// 4x3 stride-2 depthwise conv with padding 2 then 1 along H (synaptics-torq/torq-compiler-dev#2717)
module {
  func.func @main(%arg0: tensor<1x11x11x8xi8>) -> tensor<1x6x6x8xi8> {
    %w = "tosa.const"() <{values = dense<"0x38DD02F3BD64A3F9869B3E9092E6EEB7547C8EC5042A9CDBD233F0EC895B88B9EFB99F4C8362A5C842D7A9F07CEBC76CC336DD7EBA814BEA0753C3E3F78B47E216A681DB4C6CF51B8AF73822E2B3852913FE3F46B66187283F1196C54231959F"> : tensor<4x3x8x1xi8>}> : () -> tensor<4x3x8x1xi8>
    %b = "tosa.const"() <{values = dense<"0xA300000040030000D701000094FEFFFF4F02000036010000B1000000C8020000"> : tensor<8xi32>}> : () -> tensor<8xi32>
    %izp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %wzp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %m = "tosa.const"() <{values = dense<"0x194F5B48222A8F5F5F005C4565E1746F65E9AC546AEB78719E1F5D7B2211CB76"> : tensor<8xi32>}> : () -> tensor<8xi32>
    %sh = "tosa.const"() <{values = dense<"0x2626262626262626"> : tensor<8xi8>}> : () -> tensor<8xi8>
    %rzp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %ozp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.depthwise_conv2d %arg0, %w, %b, %izp, %wzp {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 2, 1, 1, 1>, stride = array<i64: 2, 2>} : (tensor<1x11x11x8xi8>, tensor<4x3x8x1xi8>, tensor<8xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x6x6x8xi32>
    %1 = tosa.rescale %0, %m, %sh, %rzp, %ozp {input_unsigned = false, output_unsigned = false, per_channel = true, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x6x6x8xi32>, tensor<8xi32>, tensor<8xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x6x6x8xi8>
    return %1 : tensor<1x6x6x8xi8>
  }
}
