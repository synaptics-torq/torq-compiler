module {
  func.func @part46_graph(%arg0: !torch.vtensor<[1,2],bf16>) -> !torch.vtensor<[1,2],bf16> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.opset_versions = {ai.onnx.ml = 2 : si64}, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Softmax"(%arg0) : (!torch.vtensor<[1,2],bf16>) -> !torch.vtensor<[1,2],bf16>
    return %0 : !torch.vtensor<[1,2],bf16>
  }
}
