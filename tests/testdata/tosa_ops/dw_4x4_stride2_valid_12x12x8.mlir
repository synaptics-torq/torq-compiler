// 4x4 stride-2 depthwise conv with no padding (synaptics-torq/torq-compiler-dev#2717)
module {
  func.func @main(%arg0: tensor<1x12x12x8xi8>) -> tensor<1x5x5x8xi8> {
    %w = "tosa.const"() <{values = dense<"0x3BAC0BE036AAA854CF6D1CE6D1C6FF48E06C83D158D0C7BD849F2B49C2C5E15FEC7BCC9411D07D8E2D68562A05438BD6372359119336F8A69623AB5B52F0C9F32ECA9E64C9675A4AECD51033A8DE8BBB073CDA4319C4D18D2A36701018340FE73B95A47EB93F50B25FF2CF71F216BB979BBB071E49280F13A2C2A16508D9F7CC"> : tensor<4x4x8x1xi8>}> : () -> tensor<4x4x8x1xi8>
    %b = "tosa.const"() <{values = dense<"0x24030000570100000C030000A7FDFFFF040200007403000079FCFFFFF2FEFFFF"> : tensor<8xi32>}> : () -> tensor<8xi32>
    %izp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %wzp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %m = "tosa.const"() <{values = dense<"0x7AA0BC68456FC04621E1A460D54E4368A22BE5700B80567B2C913A5AA8232F5C"> : tensor<8xi32>}> : () -> tensor<8xi32>
    %sh = "tosa.const"() <{values = dense<"0x2626262626262626"> : tensor<8xi8>}> : () -> tensor<8xi8>
    %rzp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %ozp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.depthwise_conv2d %arg0, %w, %b, %izp, %wzp {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x12x12x8xi8>, tensor<4x4x8x1xi8>, tensor<8xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x5x5x8xi32>
    %1 = tosa.rescale %0, %m, %sh, %rzp, %ozp {input_unsigned = false, output_unsigned = false, per_channel = true, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x5x5x8xi32>, tensor<8xi32>, tensor<8xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x5x5x8xi8>
    return %1 : tensor<1x5x5x8xi8>
  }
}
