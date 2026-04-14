module {
  func.func @part38_graph(%arg0: !torch.vtensor<[1,32,128],bf16>) -> !torch.vtensor<[1,32,64],bf16> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_model.cnn_1d_block3.depthwise_conv.conv.bias_bf16_part38_init0> : tensor<32xbf16>} : () -> !torch.vtensor<[32],bf16> 
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_model.cnn_1d_block3.depthwise_conv.conv.weight_bf16_part38_init1> : tensor<32x1x3xbf16>} : () -> !torch.vtensor<[32,1,3],bf16> 
    %none = torch.constant.none
    %2 = torch.operator "onnx.Conv"(%arg0, %1, %0) {torch.onnx.dilations = [1 : si64], torch.onnx.group = 32 : si64, torch.onnx.kernel_shape = [3 : si64], torch.onnx.pads = [0 : si64, 1 : si64], torch.onnx.strides = [2 : si64]} : (!torch.vtensor<[1,32,128],bf16>, !torch.vtensor<[32,1,3],bf16>, !torch.vtensor<[32],bf16>) -> !torch.vtensor<[1,32,64],bf16> 
    return %2 : !torch.vtensor<[1,32,64],bf16>
  }
}

{-#
  dialect_resources: {
    builtin: {
      _model.cnn_1d_block3.depthwise_conv.conv.bias_bf16_part38_init0: "0x080000008A3F0240A5BEA33F8B3F38BEE63FB63FF33C1B3D1BBF563F65BEEEBE3D3E74BD54BE953C8F3EF5BE1E3F203D92BE29BFD33D22BF40BFAB3F333E01BF243FB23E",
      _model.cnn_1d_block3.depthwise_conv.conv.weight_bf16_part38_init1: "0x080000004DBF76BF4DBC84BF60BF8CBF533F883FB3BFB4BF5DBF223FDABE3BBF13BFF33F4DC0BB3F6CBFB6BF40BF74BF84BFDFBE883FB03D98BF843F91BF873D2D3D513F143F033D4DBD60BF993F35BFA13EA23EDF3E7C3F93BF433F393E973FB7BEEC3E87BF3F3EE93F163E583F8BBF83BFBABE853FFD3EAF3F84BE32C0433D004089BFB03FC0BF95BD863F97BEB33EE73EBD3F5FBF28BD723F7BBCDC3F433D8E3F5B3F793D83BF8CBFFBBC41BFF3BE4E3FC33F8A3E883D743FDEBEA2BF553F96BF1BBF"
    }
  }
#-}

