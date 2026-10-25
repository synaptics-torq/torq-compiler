// 3x5 stride-2 conv with same padding: the kernel borders differ per axis
module {
  func.func @main(%arg0: tensor<1x11x11x4xi8>) -> tensor<1x6x6x8xi8> {
    %w = "tosa.const"() <{values = dense<"0xE8D4AE842B0344F59D193E540003794041CAD1E88198D7FF9E7713EBF1000651FF70BAEEC0A948CD3E63933F61A6EAD589A8C8079D873B2B3C503E2AF58D35E979286B2EF61A6D90A8C5AF1CB9EC9D02DC61A1963B1F39249BE7771126EB6C5657912A4182497662A18B5E594942831BCE059ED703975C89F59595A018AF7AC3564D526F388B74DF98BEDC76E595A6E0BED10331B37D77E4C682DE8B505162B54EC2E2D6970651083644BA93FD5FFA0DB32BD32949FCBBD09606632B8BC84C683FA378F8EC6BD91C10778F9DA1C47DC5AAED761976080AD4B72757BFD8F1749195A61BA6D76D5044A8307267DE8A97DAF64F8E41E43A637669AE2A0FC71C171DED5A72C2879B8DEA6CFA489E37A309EACF32889884CD7CDD972E9389413AA2BBA65AA7C40209019165B245586CF17415C4D2476990368A2FFE3810E056E6D6681E2D179291C31BF1256131A4D86CB468B9448EA8EEBE880F5EB257CF7616AE9CA48FED6D0EBC1B7041B172C4C2FF684BC527F55DBDDB1028B56537DB62AE1276B7B01351B9F0899DA0B450619F8437C946E38B4FC984CECE4CC2A98C15DFC33F5A895F570C4CD51F4214B4C94E4C681E8F9CDAFB0EA59AE4F4837854C9CCD252F3A01232E961CE67FDBEE0F050C818D352BE39E21F01EC5434449C0574F6FE8C"> : tensor<8x3x5x4xi8>}> : () -> tensor<8x3x5x4xi8>
    %b = "tosa.const"() <{values = dense<"0xFAFEFFFFBCFCFFFF69000000E6FDFFFFBB0000004BFCFFFFB802000079030000"> : tensor<8xi32>}> : () -> tensor<8xi32>
    %izp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %wzp = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %m = "tosa.const"() <{values = dense<"0x3D704F49A4F96C74ED43045A18F1AA4282C43C7A0FEC377C9F9BC14282ADF965"> : tensor<8xi32>}> : () -> tensor<8xi32>
    %sh = "tosa.const"() <{values = dense<"0x2626262626262626"> : tensor<8xi8>}> : () -> tensor<8xi8>
    %rzp = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %ozp = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %0 = tosa.conv2d %arg0, %w, %b, %izp, %wzp {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 2, 2>, stride = array<i64: 2, 2>} : (tensor<1x11x11x4xi8>, tensor<8x3x5x4xi8>, tensor<8xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x6x6x8xi32>
    %1 = tosa.rescale %0, %m, %sh, %rzp, %ozp {input_unsigned = false, output_unsigned = false, per_channel = true, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x6x6x8xi32>, tensor<8xi32>, tensor<8xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x6x6x8xi8>
    return %1 : tensor<1x6x6x8xi8>
  }
}
