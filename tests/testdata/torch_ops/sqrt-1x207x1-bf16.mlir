module  {
  func.func @main(%arg0: !torch.vtensor<[1,207,1],bf16>) -> !torch.vtensor<[1,207,1],bf16> attributes {torch.onnx_meta.ir_version = 11 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "2.1.1"} {
    %0 = torch.operator "onnx.Sqrt"(%arg0) : (!torch.vtensor<[1,207,1],bf16>) -> !torch.vtensor<[1,207,1],bf16>
    return %0 : !torch.vtensor<[1,207,1],bf16>
  }
}
