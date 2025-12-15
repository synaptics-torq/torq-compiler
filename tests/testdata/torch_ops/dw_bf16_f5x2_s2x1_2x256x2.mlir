module  {
  func.func @main(%7817: !torch.vtensor<[1,2,256,2],bf16>) -> !torch.vtensor<[1,2,128,1],bf16> attributes {torch.onnx_meta.ir_version = 11 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "2.1.1"} {
    %3 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_encoder.0.1.conv.depth_conv2d.weight_bf16> : tensor<2x1x5x2xbf16>} : () -> !torch.vtensor<[2,1,5,2],bf16>
    %4 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_encoder.0.1.conv.depth_conv2d.bias_bf16> : tensor<2xbf16>} : () -> !torch.vtensor<[2],bf16>
    %7818 = torch.operator "onnx.Conv"(%7817, %3, %4) {torch.onnx.dilations = [1 : si64, 1 : si64], torch.onnx.group = 2 : si64, torch.onnx.kernel_shape = [5 : si64, 2 : si64], torch.onnx.pads = [2 : si64, 0 : si64, 2 : si64, 0 : si64], torch.onnx.strides = [2 : si64, 1 : si64]} : (!torch.vtensor<[1,2,256,2],bf16>, !torch.vtensor<[2,1,5,2],bf16>, !torch.vtensor<[2],bf16>) -> !torch.vtensor<[1,2,128,1],bf16>
    return %7818 : !torch.vtensor<[1,2,128,1],bf16>
  }
}

{-#
  dialect_resources: {
    builtin: {
      _encoder.0.1.conv.depth_conv2d.weight_bf16: "0x0800000053BD80BDBFBC83BC1DBDD3BB913D87BC133C4ABD1CBDC4BD043C34BD0EBBC13C86BD7E3D9A3C673D",
      _encoder.0.1.conv.depth_conv2d.bias_bf16: "0x0800000000000000"
    }
  }
#-}
