// 3x5 stride-2 depthwise conv with same padding: the kernel borders differ per axis
module {
  func.func @main(%arg0: tensor<1x11x11x8xi8>) -> tensor<1x6x6x8xi8> {
    %w = "tosa.const"() <{values = dense<"0xDF0141584E6A8F22A1565801773C90C537674CCE28BDFD8A9D6FC29335A7BB84BBD15DACF8FAB24FCE9CC025F9D92968FF316CF1FBCC1AC5190177DFC93F393EFF5B206681AB0A6C3ED60E8767E06ECCB7C7CD465A4F2F0F3D2AE8F8326F7465E43E57FE5A9F14660CF3901888F9524E428732493925A738"> : tensor<3x5x8x1xi8>}> : () -> tensor<3x5x8x1xi8>
    %b = "tosa.const"() <{values = dense<"0x77FDFFFFCAFCFFFFD602000044FCFFFF52000000B8FCFFFF6FFEFFFFDAFFFFFF"> : tensor<8xi32>}> : () -> tensor<8xi32>
    %izp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %wzp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %m = "tosa.const"() <{values = dense<"0xB34E0D5BC2A8CE594FFDCF4199B25740D741F447E9BA8740A982EB6AAEB8A361"> : tensor<8xi32>}> : () -> tensor<8xi32>
    %sh = "tosa.const"() <{values = dense<"0x2626262626262626"> : tensor<8xi8>}> : () -> tensor<8xi8>
    %rzp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %ozp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.depthwise_conv2d %arg0, %w, %b, %izp, %wzp {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 2, 2>, stride = array<i64: 2, 2>} : (tensor<1x11x11x8xi8>, tensor<3x5x8x1xi8>, tensor<8xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x6x6x8xi32>
    %1 = tosa.rescale %0, %m, %sh, %rzp, %ozp {input_unsigned = false, output_unsigned = false, per_channel = true, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x6x6x8xi32>, tensor<8xi32>, tensor<8xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x6x6x8xi8>
    return %1 : tensor<1x6x6x8xi8>
  }
}
