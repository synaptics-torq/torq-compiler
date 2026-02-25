module {
  func.func @maxpool_test_batch1_3x3_stride1x1_16x4(%arg0: !torch.vtensor<[1,16,4,32],bf16>) -> !torch.vtensor<[1,16,4,32],bf16> attributes {torch.onnx_meta.ir_version = 11 : si64, torch.onnx_meta.opset_version = 23 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.MaxPool"(%arg0) {torch.onnx.ceil_mode = 0 : si64, torch.onnx.dilations = [1 : si64, 1 : si64], torch.onnx.kernel_shape = [3 : si64, 3 : si64], torch.onnx.pads = [1 : si64, 1 : si64, 1 : si64, 1 : si64], torch.onnx.strides = [1 : si64, 1 : si64]} : (!torch.vtensor<[1,16,4,32],bf16>) -> !torch.vtensor<[1,16,4,32],bf16>
    return %0 : !torch.vtensor<[1,16,4,32],bf16>
  }
}
