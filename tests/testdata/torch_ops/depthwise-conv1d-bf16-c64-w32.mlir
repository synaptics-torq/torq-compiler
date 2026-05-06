module {
  func.func @part9_graph(%arg0: !torch.vtensor<[1,64,32],bf16>) -> !torch.vtensor<[1,64,32],bf16> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__net_cnn_1d_block4_depthwise_conv_activation_Constant_1_output_0_bf16_part9_init0> : tensor<bf16>} : () -> !torch.vtensor<[],bf16> 
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__net_cnn_1d_block4_depthwise_conv_activation_Constant_output_0_bf16_part9_init1> : tensor<bf16>} : () -> !torch.vtensor<[],bf16> 
    %2 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_net.cnn_1d_block4.depthwise_conv.conv.bias_bf16_part9_init2> : tensor<64xbf16>} : () -> !torch.vtensor<[64],bf16> 
    %3 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_net.cnn_1d_block4.depthwise_conv.conv.weight_bf16_part9_init3> : tensor<64x1x3xbf16>} : () -> !torch.vtensor<[64,1,3],bf16> 
    %none = torch.constant.none
    %4 = torch.operator "onnx.Conv"(%arg0, %3, %2) {torch.onnx.dilations = [1 : si64], torch.onnx.group = 64 : si64, torch.onnx.kernel_shape = [3 : si64], torch.onnx.pads = [1 : si64, 1 : si64], torch.onnx.strides = [1 : si64]} : (!torch.vtensor<[1,64,32],bf16>, !torch.vtensor<[64,1,3],bf16>, !torch.vtensor<[64],bf16>) -> !torch.vtensor<[1,64,32],bf16> 
    %5 = torch.operator "onnx.Clip"(%4, %1, %0) : (!torch.vtensor<[1,64,32],bf16>, !torch.vtensor<[],bf16>, !torch.vtensor<[],bf16>) -> !torch.vtensor<[1,64,32],bf16> 
    return %5 : !torch.vtensor<[1,64,32],bf16>
  }
}

{-#
  dialect_resources: {
    builtin: {
      __net_cnn_1d_block4_depthwise_conv_activation_Constant_1_output_0_bf16_part9_init0: "0x08000000C040",
      __net_cnn_1d_block4_depthwise_conv_activation_Constant_output_0_bf16_part9_init1: "0x080000000000",
      _net.cnn_1d_block4.depthwise_conv.conv.bias_bf16_part9_init2: "0x080000006CBF30BFC73E0ABED6BD24BFE13D70BD9E3EF9BC06BE43BF1FBF30BD05BD333FA83E23BF1E3F26BF20BEC3BE833F81BF38BE58BF3A3FB53E0DBF843F33BFDBBE2DBC223FBC3EBABC5FBF3E3F13BF83BFE8BC85BF2EBE043E03BF473E40BFBE3C3C3DCFBE3DBF063F4ABF1BBE9A3FD8BB74BDC9BEA73E693EB83EB03EA73FA63D",
      _net.cnn_1d_block4.depthwise_conv.conv.weight_bf16_part9_init3: "0x08000000863E7F3F853FE03E8A3F8FBE223F69BEECBF293F363FFFBF663F393F85BF663F133FAFBC793F1CBF97BFE1BE493F63BF643FA3BF8BBF583F573FAABFFA3F23BDEABF173D363F3A3F193F69BE713F93BF86BC813FB6BF803EB33F94BF75BF9E3E90BFBC3E55BF1F3D233F403F0DBF91BF28BE5D3F783F7EBEE73EB4BF453FB23F1DBFCBBE2DBF57BF0DBF0E3FCD3E7E3F8ABF793F5ABF393F5D3FF43E84BFF3BE94BDC93F8CBFA2BF8B3FDF3E753EDBBEE0BE38BF103D98BEFA3F0F3DB8BE863FB2BFD7BBB43F86BF70BFFABD9DBF52BFAB3E08C07E3F823F043FB23E7D3F5CBE17BF85BFD53D4A3E813F9D3FB63F31BF853F28BF40BF1D3F993F6B3F9D3F3C3F50BFA8BF4F3F56BF513FBA3EE03EBDBF853EA4BD3C3FB53E813E4F3F84BF8CBCA5BFC8BE6B3FF83E873FED3E753F553F22BF14BFB1BF043F913F643F7A3F883F7ABF0ABF8BBD62BFF6BF80BFE23E643F41BF813F54BE9ABF6A3FF93EACBFCD3E83BF493E4CBFB4BF05C07EBF17402BC040BEBE3FB2BEADBFA7BF813F08BF22BF"
    }
  }
#-}

