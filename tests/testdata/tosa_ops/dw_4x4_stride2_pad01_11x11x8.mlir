// 4x4 stride-2 depthwise conv with padding 0 then 1 (synaptics-torq/torq-compiler-dev#2717)
module {
  func.func @main(%arg0: tensor<1x11x11x8xi8>) -> tensor<1x5x5x8xi8> {
    %w = "tosa.const"() <{values = dense<"0x6F5F081CE59A1842B330B8E29A747DF5FDA6C97EBF44934DD686987AADF1A8E1722CFF2EBE45B97245FF03253B249E568BBE2F2EE9B9333BDA0FE339BB8A5FBE2E14A29584B31338F53237BBEAF157701DFD0598CE9BE07482ECCF988DC7EC4033DFFCC332843705F9346211252B9251F99D6E1BA08607CCF717DB5024B82BEB"> : tensor<4x4x8x1xi8>}> : () -> tensor<4x4x8x1xi8>
    %b = "tosa.const"() <{values = dense<"0x15FEFFFF9BFEFFFFB7010000BC00000008000000BBFEFFFF0902000027FFFFFF"> : tensor<8xi32>}> : () -> tensor<8xi32>
    %izp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %wzp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %m = "tosa.const"() <{values = dense<"0x7AD200554541FA78A82EE45000C0894EAE00B56D544CE267BB621B43E4816045"> : tensor<8xi32>}> : () -> tensor<8xi32>
    %sh = "tosa.const"() <{values = dense<"0x2626262626262626"> : tensor<8xi8>}> : () -> tensor<8xi8>
    %rzp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %ozp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.depthwise_conv2d %arg0, %w, %b, %izp, %wzp {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 2, 2>} : (tensor<1x11x11x8xi8>, tensor<4x4x8x1xi8>, tensor<8xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x5x5x8xi32>
    %1 = tosa.rescale %0, %m, %sh, %rzp, %ozp {input_unsigned = false, output_unsigned = false, per_channel = true, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x5x5x8xi32>, tensor<8xi32>, tensor<8xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x5x5x8xi8>
    return %1 : tensor<1x5x5x8xi8>
  }
}
