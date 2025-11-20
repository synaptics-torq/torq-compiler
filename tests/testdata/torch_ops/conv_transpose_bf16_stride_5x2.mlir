module  {
  func.func @main(%8033: !torch.vtensor<[1,1,16,2],bf16>) -> !torch.vtensor<[1,1,32,1],bf16> attributes {torch.onnx_meta.ir_version = 11 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "2.1.1"} {
    %2134 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__decoder.0_decoder.0.0_deconv_depth_ConvTranspose2d_ConvTranspose_Wg0_bf16> : tensor<1x1x5x2xbf16>} : () -> !torch.vtensor<[1,1,5,2],bf16>
    %2135 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__decoder.0_decoder.0.0_deconv_depth_ConvTranspose2d_ConvTranspose_Bg0_bf16> : tensor<1xbf16>} : () -> !torch.vtensor<[1],bf16>
    %8545 = torch.operator "onnx.ConvTranspose"(%8033, %2134, %2135) {torch.onnx.dilations = [1 : si64, 1 : si64], torch.onnx.group = 1 : si64, torch.onnx.kernel_shape = [5 : si64, 2 : si64], torch.onnx.output_padding = [1 : si64, 0 : si64], torch.onnx.pads = [2 : si64, 1 : si64, 2 : si64, 1 : si64], torch.onnx.strides = [2 : si64, 1 : si64]} : (!torch.vtensor<[1,1,16,2],bf16>, !torch.vtensor<[1,1,5,2],bf16>, !torch.vtensor<[1],bf16>) -> !torch.vtensor<[1,1,32,1],bf16>
    return %8545 : !torch.vtensor<[1,1,32,1],bf16>
  }
}

{-#
  dialect_resources: {
    builtin: {
      __decoder.0_decoder.0.0_deconv_depth_ConvTranspose2d_ConvTranspose_Wg0_bf16: "0x080000002FBD983DCCBBC7BC0B3C7FBD683D8E3CEABB91BC",
      __decoder.0_decoder.0.0_deconv_depth_ConvTranspose2d_ConvTranspose_Bg0_bf16: "0x080000000000"
    }
  }
#-}