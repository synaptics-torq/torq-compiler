module {
  func.func @part16_graph(%arg0: !torch.vtensor<[1,16,33,20],bf16>) -> !torch.vtensor<[1,16,16,10],bf16> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.opset_versions = {ai.onnx.ml = 2 : si64}, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.MaxPool"(%arg0) {torch.onnx.kernel_shape = [2 : si64, 2 : si64], torch.onnx.strides = [2 : si64, 2 : si64]} : (!torch.vtensor<[1,16,33,20],bf16>) -> !torch.vtensor<[1,16,16,10],bf16> 
    return %0 : !torch.vtensor<[1,16,16,10],bf16>
  }
}

