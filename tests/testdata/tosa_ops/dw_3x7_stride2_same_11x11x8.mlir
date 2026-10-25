// 3x7 stride-2 depthwise conv with same padding: the kernel borders differ per axis
module {
  func.func @main(%arg0: tensor<1x11x11x8xi8>) -> tensor<1x6x6x8xi8> {
    %w = "tosa.const"() <{values = dense<"0xE97AC611B28FF6F2B583238103BBA70A09D64B443BCA533CAF1EEF79220AF1065F378116BA0EBF765B2655D1AD16EA4BF200B0002B25DC1933AA2BCC64E113760D71B1B2E14B270CDA62123854226A3DD8AC19B92BE6619DDCE175EBF6EB967778FF92D614258EB87DFF76C2950DA2ACFD3DC69739D0CE3ADEAD03759B2DE1D7D8E369AF826DDC6A99908D220C348AD45238388EE2C54A73F796A2B8F8D16D39A4E55B8A3A795BAD"> : tensor<3x7x8x1xi8>}> : () -> tensor<3x7x8x1xi8>
    %b = "tosa.const"() <{values = dense<"0xE6FDFFFFC8FFFFFFC70000003903000026FDFFFF13020000520300003E030000"> : tensor<8xi32>}> : () -> tensor<8xi32>
    %izp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %wzp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %m = "tosa.const"() <{values = dense<"0xC01ABB51F35E2748CAF600662F41B54460D75148AF3980443E4B7F4D094F9B77"> : tensor<8xi32>}> : () -> tensor<8xi32>
    %sh = "tosa.const"() <{values = dense<"0x2626262626262626"> : tensor<8xi8>}> : () -> tensor<8xi8>
    %rzp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %ozp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.depthwise_conv2d %arg0, %w, %b, %izp, %wzp {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 3, 3>, stride = array<i64: 2, 2>} : (tensor<1x11x11x8xi8>, tensor<3x7x8x1xi8>, tensor<8xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x6x6x8xi32>
    %1 = tosa.rescale %0, %m, %sh, %rzp, %ozp {input_unsigned = false, output_unsigned = false, per_channel = true, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x6x6x8xi32>, tensor<8xi32>, tensor<8xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x6x6x8xi8>
    return %1 : tensor<1x6x6x8xi8>
  }
}
