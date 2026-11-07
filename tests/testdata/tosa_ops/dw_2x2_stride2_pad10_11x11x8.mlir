// 2x2 stride-2 depthwise conv with padding 1 then 0 (synaptics-torq/torq-compiler-dev#2717)
module {
  func.func @main(%arg0: tensor<1x11x11x8xi8>) -> tensor<1x6x6x8xi8> {
    %w = "tosa.const"() <{values = dense<"0x549361119303F4109F1C7B447E2B611F9AD2651F2688BA645FF2C14176DC03AB"> : tensor<2x2x8x1xi8>}> : () -> tensor<2x2x8x1xi8>
    %b = "tosa.const"() <{values = dense<"0xA603000043FDFFFF790300000BFDFFFFC4030000B0FCFFFF0503000044000000"> : tensor<8xi32>}> : () -> tensor<8xi32>
    %izp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %wzp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %m = "tosa.const"() <{values = dense<"0x15010F64D5569B4AFC5E6B48ACA3A8738BAA9E4F70737241409C1B5D848FF957"> : tensor<8xi32>}> : () -> tensor<8xi32>
    %sh = "tosa.const"() <{values = dense<"0x2626262626262626"> : tensor<8xi8>}> : () -> tensor<8xi8>
    %rzp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %ozp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.depthwise_conv2d %arg0, %w, %b, %izp, %wzp pad([1, 0, 1, 0]) stride([2, 2]) dilation([1, 1]) acc_type(i32) : (tensor<1x11x11x8xi8>, tensor<2x2x8x1xi8>, tensor<8xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x6x6x8xi32>
    %1 = tosa.rescale %0, %m, %sh, %rzp, %ozp scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(true) input_unsigned(false) output_unsigned(false) : (tensor<1x6x6x8xi32>, tensor<8xi32>, tensor<8xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x6x6x8xi8>
    return %1 : tensor<1x6x6x8xi8>
  }
}
