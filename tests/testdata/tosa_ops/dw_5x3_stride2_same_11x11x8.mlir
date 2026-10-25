// 5x3 stride-2 depthwise conv with same padding: the kernel borders differ per axis
module {
  func.func @main(%arg0: tensor<1x11x11x8xi8>) -> tensor<1x6x6x8xi8> {
    %w = "tosa.const"() <{values = dense<"0x9623AB5B52F0C9F32ECA9E64C9675A4AECD51033A8DE8BBB073CDA4319C4D18D2A36701018340FE73B95A47EB93F50B25FF2CF71F216BB979BBB071E49280F13A2C2A16508D9F7CC94E16D65B75A872B6D8A706231F193B363A89C41A6D3AE709455F08CA85BF7DD6B007121973C819B06031201D8BB8D20"> : tensor<5x3x8x1xi8>}> : () -> tensor<5x3x8x1xi8>
    %b = "tosa.const"() <{values = dense<"0x0F020000560300004BFFFFFF88FFFFFFCCFFFFFF8D0300009FFDFFFFFFFFFFFF"> : tensor<8xi32>}> : () -> tensor<8xi32>
    %izp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %wzp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %m = "tosa.const"() <{values = dense<"0x9B7530431EF2365BAB03807CC293B167D90360563FA9AF7FD0F4A3663F7EBB7C"> : tensor<8xi32>}> : () -> tensor<8xi32>
    %sh = "tosa.const"() <{values = dense<"0x2626262626262626"> : tensor<8xi8>}> : () -> tensor<8xi8>
    %rzp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %ozp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.depthwise_conv2d %arg0, %w, %b, %izp, %wzp {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 2, 2, 1, 1>, stride = array<i64: 2, 2>} : (tensor<1x11x11x8xi8>, tensor<5x3x8x1xi8>, tensor<8xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x6x6x8xi32>
    %1 = tosa.rescale %0, %m, %sh, %rzp, %ozp {input_unsigned = false, output_unsigned = false, per_channel = true, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x6x6x8xi32>, tensor<8xi32>, tensor<8xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x6x6x8xi8>
    return %1 : tensor<1x6x6x8xi8>
  }
}
