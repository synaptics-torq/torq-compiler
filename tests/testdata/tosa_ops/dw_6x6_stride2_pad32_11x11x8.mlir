// 6x6 stride-2 depthwise conv with padding 3 then 2 (synaptics-torq/torq-compiler-dev#2717)
module {
  func.func @main(%arg0: tensor<1x11x11x8xi8>) -> tensor<1x6x6x8xi8> {
    %w = "tosa.const"() <{values = dense<"0x52F920BF52CAED2ADEAD6E5EB24F2A24A40FF0065D1DB46D468745DF6AB116CEF8BDADA8E926CBEF6FAE38FE6F68205B40CBCD78F2FA09148C123E2246AC5A471AA4DD190EF985D30383C878351949FE93EFE73CB1BD15DA9F42EFE202D3BDE9B91C0D3827B542C17AD26B86E01FA99A1C364B216C2226A689FCBE54EFC094D25E7E3801C268268229456D8122CA9BC323532A52A0138F4BF1C204A47B46FC4C7BA6DA0CDB133A1385B30B4A64922C6927EE626CEBF2E7B49C02E129FE81906A50B14A74889563F52946344F2C8C8F00D15D8B7711B6663264D00760B6D644FE900133BA03DD976EA8179EC25CE027456C3742FAD37CB8403FA5E3340D9EF7AA66C74B2173B3D30B8F3AF9533138AF8DF510B65EB87D7557E2656F99D67AF957"> : tensor<6x6x8x1xi8>}> : () -> tensor<6x6x8x1xi8>
    %b = "tosa.const"() <{values = dense<"0x6B00000045FFFFFF4CFEFFFF7DFEFFFFE7010000E6FDFFFF3DFDFFFF2D010000"> : tensor<8xi32>}> : () -> tensor<8xi32>
    %izp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %wzp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %m = "tosa.const"() <{values = dense<"0x37F965628D9DF0501BA26853A9852F7726679A42A54D5251B099C063FD52186B"> : tensor<8xi32>}> : () -> tensor<8xi32>
    %sh = "tosa.const"() <{values = dense<"0x2626262626262626"> : tensor<8xi8>}> : () -> tensor<8xi8>
    %rzp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %ozp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.depthwise_conv2d %arg0, %w, %b, %izp, %wzp pad([3, 2, 3, 2]) stride([2, 2]) dilation([1, 1]) acc_type(i32) : (tensor<1x11x11x8xi8>, tensor<6x6x8x1xi8>, tensor<8xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x6x6x8xi32>
    %1 = tosa.rescale %0, %m, %sh, %rzp, %ozp scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(true) input_unsigned(false) output_unsigned(false) : (tensor<1x6x6x8xi32>, tensor<8xi32>, tensor<8xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x6x6x8xi8>
    return %1 : tensor<1x6x6x8xi8>
  }
}
