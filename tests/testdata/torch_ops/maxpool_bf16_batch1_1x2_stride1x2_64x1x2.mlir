module {
  func.func @part39_graph(%arg0: !torch.vtensor<[1,64,1,2],bf16>) -> !torch.vtensor<[1,64,1,1],bf16> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.opset_versions = {ai.onnx.ml = 2 : si64}, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.MaxPool"(%arg0) {torch.onnx.kernel_shape = [1 : si64, 2 : si64], torch.onnx.strides = [1 : si64, 2 : si64]} : (!torch.vtensor<[1,64,1,2],bf16>) -> !torch.vtensor<[1,64,1,1],bf16>
    return %0 : !torch.vtensor<[1,64,1,1],bf16>
  }
}
