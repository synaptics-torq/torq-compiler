// 4x4 stride-2 depthwise conv with padding 1 then 0 (synaptics-torq/torq-compiler-dev#2717)
module {
  func.func @main(%arg0: tensor<1x11x11x8xi8>) -> tensor<1x5x5x8xi8> {
    %w = "tosa.const"() <{values = dense<"0xB066E5F9668AE073B3619CB28BAC78FFF055408CFD475AEC308E718CCF451DE88FD982243D7DC3520E1981786C71E8D4AE842B0344F59D193E540003794041CAD1E88198D7FF9E7713EBF1000651FF70BAEEC0A948CD3E63933F61A6EAD589A8C8079D873B2B3C503E2AF58D35E979286B2EF61A6D90A8C5AF1CB9EC9D02DC61"> : tensor<4x4x8x1xi8>}> : () -> tensor<4x4x8x1xi8>
    %b = "tosa.const"() <{values = dense<"0xCA010000AC0300005603000050010000A7030000E602000035FCFFFF06FDFFFF"> : tensor<8xi32>}> : () -> tensor<8xi32>
    %izp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %wzp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %m = "tosa.const"() <{values = dense<"0x14E14577425D454547E6CB7ECFB5F3747BEE427D17860F577D598549974F1461"> : tensor<8xi32>}> : () -> tensor<8xi32>
    %sh = "tosa.const"() <{values = dense<"0x2626262626262626"> : tensor<8xi8>}> : () -> tensor<8xi8>
    %rzp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %ozp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.depthwise_conv2d %arg0, %w, %b, %izp, %wzp {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 0, 1, 0>, stride = array<i64: 2, 2>} : (tensor<1x11x11x8xi8>, tensor<4x4x8x1xi8>, tensor<8xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x5x5x8xi32>
    %1 = tosa.rescale %0, %m, %sh, %rzp, %ozp {input_unsigned = false, output_unsigned = false, per_channel = true, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x5x5x8xi32>, tensor<8xi32>, tensor<8xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x5x5x8xi8>
    return %1 : tensor<1x5x5x8xi8>
  }
}
