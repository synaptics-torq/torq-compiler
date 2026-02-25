module {
  func.func @maxpool_test_batch19_2x2_stride2x2_32x64x16(%arg0: !torch.vtensor<[19,32,64,16],bf16>) -> !torch.vtensor<[19,32,32,8],bf16> attributes {torch.onnx_meta.ir_version = 11 : si64, torch.onnx_meta.opset_version = 23 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.MaxPool"(%arg0) {torch.onnx.ceil_mode = 0 : si64, torch.onnx.dilations = [1 : si64, 1 : si64], torch.onnx.kernel_shape = [2 : si64, 2 : si64], torch.onnx.pads = [0 : si64, 0 : si64, 0 : si64, 0 : si64], torch.onnx.strides = [2 : si64, 2 : si64]} : (!torch.vtensor<[19,32,64,16],bf16>) -> !torch.vtensor<[19,32,32,8],bf16>
    return %0 : !torch.vtensor<[19,32,32,8],bf16>
  }
}
