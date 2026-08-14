module {
  func.func @main_graph(%arg0: !torch.vtensor<[1,3,241,320],bf16>) -> !torch.vtensor<[1,3,121,160],bf16> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.MaxPool"(%arg0) {torch.onnx.ceil_mode = 0 : si64, torch.onnx.kernel_shape = [3 : si64, 3 : si64], torch.onnx.pads = [1 : si64, 1 : si64, 1 : si64, 1 : si64], torch.onnx.strides = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,3,241,320],bf16>) -> !torch.vtensor<[1,3,121,160],bf16>
    return %0 : !torch.vtensor<[1,3,121,160],bf16>
  }
}
