module attributes {tfl.description = "Extracted layer range", tfl.schema_version = 3 : i32} {
  func.func @main(%arg0: tensor<1x2x64x100xi8>, %arg1: tensor<1x2x100x100xi8>, %arg2: tensor<1x3x3x128xi8>) -> tensor<1x128x10x10xi8> attributes {tf.entry_function = {inputs = "ultralytics.utils.export.engine._NormalizeCoords/ultralytics.nn.tasks.DetectionModel_model/ultralytics.nn.modules.block.C2PSA_10/torch.nn.modules.container.Sequential_m/ultralytics.nn.modules.block.PSABlock_0/ultralytics.nn.modules.block.Attention_attn;4,ultralytics.utils.export.engine._NormalizeCoords/ultralytics.nn.tasks.DetectionModel_model/ultralytics.nn.modules.block.C2PSA_10/torch.nn.modules.container.Sequential_m/ultralytics.nn.modules.block.PSABlock_0/ultralytics.nn.modules.block.Attention_attn;12,ultralytics.utils.export.engine._NormalizeCoords/ultralytics.nn.tasks.DetectionModel_model/ultralytics.nn.modules.block.C2PSA_10/torch.nn.modules.container.Sequential_m/ultralytics.nn.modules.block.PSABlock_0/ultralytics.nn.modules.block.Attention_attn/ultralytics.nn.modules.conv.Conv_pe/torch.nn.modules.conv.Conv2d_conv;1", outputs = "ultralytics.utils.export.engine._NormalizeCoords/ultralytics.nn.tasks.DetectionModel_model/ultralytics.nn.modules.block.C2PSA_10/torch.nn.modules.container.Sequential_m/ultralytics.nn.modules.block.PSABlock_0/ultralytics.nn.modules.block.Attention_attn;18"}} {
    %0 = tosa.const_shape  {values = dense<[3, 3, 128, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %1 = "tosa.const"() <{values = dense<19> : tensor<1xi8>}> : () -> tensor<1xi8>
    %2 = "tosa.const"() <{values = dense<49> : tensor<1xi8>}> : () -> tensor<1xi8>
    %3 = "tosa.const"() <{values = dense<2010547540> : tensor<1xi32>}> : () -> tensor<1xi32>
    %4 = "tosa.const"() <{values = dense<32> : tensor<1xi8>}> : () -> tensor<1xi8>
    %5 = "tosa.const"() <{values = dense<1832923658> : tensor<1xi32>}> : () -> tensor<1xi32>
    %6 = "tosa.const"() <{values = dense<10> : tensor<1xi8>}> : () -> tensor<1xi8>
    %7 = "tosa.const"() <{values = dense<11> : tensor<1xi8>}> : () -> tensor<1xi8>
    %8 = "tosa.const"() <{values = dense<1073741824> : tensor<1xi32>}> : () -> tensor<1xi32>
    %9 = "tosa.const"() <{values = dense<"0x2626262525262626262626242626262526272626262526262626262625262626262526262726262626252626262526262526262626262626262626262726262526252826272524272427272727272727272727242727272727272727262726262626272627272727272526272728272727262627272724242524242627272727"> : tensor<128xi8>}> : () -> tensor<128xi8>
    %10 = "tosa.const"() <{values = dense<"0x648EF8686F31127658866E7D088402435B445B449C9C5177AD5BA276C78D836CB5061765E9FCCB6606B48440E63B375C681FE475F179FA493FD7F3737B3CC2442C08F367C29C54450390D7720FDF8873070C715D56BB9540B2E4B17C0F41AC6EADD7C56ACC93547C51B5807091249470F7F72E405436F57F1DF2CF7351193C6F8866605C8F6CD24605E46775C201CF6FDF8AC564147771740E998E71DF91CD74C2EEB570452111430E27F4784FD47E7D32144575EB0CEE45FA6C7F6E23B96A79B5E4B2405887316E1D702F6D742EE66A2934687357980D7282D0B07E94B8D97BD9735D7208B601702D30536D62BD0B5997E58B6C881F6B767BDA2D6DA71FB441F4A82D74E7CFC444C177F35BAECB6E64A3009A6FD9324562BFE009636B5F7A67F86154725B14ED73DDECD35B6B074C767757C35CB9153F696E4D0E53CCDCBC642156815F4FD2436E7B273175B75880669060CB71DF22C76CA901336633990B72DE40D162C90C85667ACDD97099233B774ACB435FCA260768EE18F04086C86741E7B11544FB2DEE6E3BCD1A53A9516F439F101E4FEF01DE5D132CE065D8D160760530894FDF64B94A75404147B4E6EB500C6CF35DAFE5705C5C340070D4DB9A68A269D87F43FD9456665D9148EDDB917ABB692D693F11CD63B58DBE5EF129746A730EC268A758C854DC6D705A4BA7B5443416AB5CAF760C7DFD9C0A71C137C06E"> : tensor<128xi32>}> : () -> tensor<128xi32>
    %11 = "tosa.const"() <{values = dense<1> : tensor<1xi8>}> : () -> tensor<1xi8>
    %12 = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %13 = "tosa.const"() <{values = dense<"0xBAFEFFFF6CFCFFFFDAFFFFFF9DFFFFFFD2FFFFFFB4FFFFFF5CFFFFFFDAFFFFFF9FFEFFFFAF000000D2FAFFFFFEFFFFFF70010000F8090000EC00000000FFFFFF88FFFFFFFE000000CBFEFFFF08000000A101000088FEFFFF32FFFFFF8C000000CCFBFFFF500000004D00000096FDFFFF6EFEFFFFD6FEFFFFC1FFFFFF16FFFFFF39000000CFFFFFFF4A010000A1FFFFFFE10300004D010000F6000000D000000063000000BAFEFFFF19000000C2000000EC000000C2FCFFFF1C0000003DFFFFFF1A0100006FFEFFFF9701000031FFFFFF36010000C5FFFFFF250300009E000000DFFEFFFFA8000000D8FFFFFFB5E9FFFF4700000016FDFFFFF4FEFFFFAF00000030FDFFFFFCFFFFFF9BFFFFFF6F000000C7FFFFFFCDFFFFFF0B000000B9FCFFFF87FFFFFFBC0000006FFCFFFF82000000710000007DFFFFFF2B0000001503000062FFFFFF95000000B7000000FBFFFFFF23FFFFFF0C020000C2FFFFFFF5FFFFFFC4FFFFFFEF0400004E010000AFFCFFFF0E000000730500004F000000B9F3FFFF6E000000B3030000EFFEFFFF9C010000AAFBFFFF42FFFFFF640000009A010000E8E1FFFF06FDFFFF54000000BA030000A0010000E0F6FFFF59000000590000002E010000B2000000D2FEFFFFCEFFFFFFD7FBFFFF0E000000FEFFFFFFDEFFFFFF0FFDFFFF26000000FBFFFFFFF4FEFFFF94FFFFFF520000006200000043FEFFFF"> : tensor<128xi32>}> : () -> tensor<128xi32>
    %14 = tosa.const_shape  {values = dense<[1, 128, 10, 10]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %15 = "tosa.const"() <{values = dense<38> : tensor<1xi8>}> : () -> tensor<1xi8>
    %16 = "tosa.const"() <{values = dense<1306670402> : tensor<1xi32>}> : () -> tensor<1xi32>
    %17 = "tosa.const"() <{values = dense<6> : tensor<1xi8>}> : () -> tensor<1xi8>
    %18 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %19 = "tosa.const"() <{values = dense<-128> : tensor<1xi8>}> : () -> tensor<1xi8>
    %20 = "tosa.const"() <{values = dense<5> : tensor<1xi8>}> : () -> tensor<1xi8>
    %21 = tosa.const_shape  {values = dense<[2, 100, 100]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %22 = tosa.const_shape  {values = dense<[2, 64, 100]> : tensor<3xindex>} : () -> !tosa.shape<3>
    %23 = tosa.reshape %arg0, %22 : (tensor<1x2x64x100xi8>, !tosa.shape<3>) -> tensor<2x64x100xi8>
    %24 = tosa.reshape %arg1, %21 : (tensor<1x2x100x100xi8>, !tosa.shape<3>) -> tensor<2x100x100xi8>
    %25 = tosa.matmul %23, %24, %20, %19 : (tensor<2x64x100xi8>, tensor<2x100x100xi8>, tensor<1xi8>, tensor<1xi8>) -> tensor<2x64x100xi32>
    %26 = tosa.rescale %25, %16, %15, %18, %17 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<2x64x100xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<2x64x100xi8>
    %27 = tosa.reshape %26, %14 : (tensor<2x64x100xi8>, !tosa.shape<4>) -> tensor<1x128x10x10xi8>
    %28 = tosa.reshape %arg0, %14 : (tensor<1x2x64x100xi8>, !tosa.shape<4>) -> tensor<1x128x10x10xi8>
    %29 = tosa.transpose %28 {perms = array<i32: 0, 2, 3, 1>} : (tensor<1x128x10x10xi8>) -> tensor<1x10x10x128xi8>
    %30 = tosa.reshape %arg2, %0 : (tensor<1x3x3x128xi8>, !tosa.shape<4>) -> tensor<3x3x128x1xi8>
    %31 = tosa.depthwise_conv2d %29, %30, %13, %20, %12 {acc_type = i32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x10x10x128xi8>, tensor<3x3x128x1xi8>, tensor<128xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x10x10x128xi32>
    %32 = tosa.rescale %31, %10, %9, %18, %11 {input_unsigned = false, output_unsigned = false, per_channel = true, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x10x10x128xi32>, tensor<128xi32>, tensor<128xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x10x10x128xi8>
    %33 = tosa.transpose %32 {perms = array<i32: 0, 3, 1, 2>} : (tensor<1x10x10x128xi8>) -> tensor<1x128x10x10xi8>
    %34 = tosa.rescale %27, %8, %7, %17, %18 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x128x10x10xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x128x10x10xi32>
    %35 = tosa.rescale %33, %8, %6, %11, %18 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x128x10x10xi8>, tensor<1xi32>, tensor<1xi8>, tensor<1xi8>, tensor<1xi32>) -> tensor<1x128x10x10xi32>
    %36 = tosa.rescale %35, %5, %4, %18, %18 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x128x10x10xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi32>) -> tensor<1x128x10x10xi32>
    %37 = tosa.add %34, %36 : (tensor<1x128x10x10xi32>, tensor<1x128x10x10xi32>) -> tensor<1x128x10x10xi32>
    %38 = tosa.rescale %37, %3, %2, %18, %1 {input_unsigned = false, output_unsigned = false, per_channel = false, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x128x10x10xi32>, tensor<1xi32>, tensor<1xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x128x10x10xi8>
    return %38 : tensor<1x128x10x10xi8>
  }
}
