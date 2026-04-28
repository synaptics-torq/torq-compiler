module {
  func.func @main(%arg0: !torch.vtensor<[1,1024,64],bf16>) -> !torch.vtensor<[1,1024,64],bf16> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.opset_versions = {ai.onnx.ml = 5 : si64, ai.onnx.preview.training = 1 : si64, ai.onnx.training = 1 : si64, com.microsoft = 1 : si64, com.microsoft.experimental = 1 : si64, com.microsoft.nchwc = 1 : si64, org.pytorch.aten = 1 : si64}, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    // TORQ_ALLOWED_WRONG: .01
    %0 = torch.operator "onnx.Gelu"(%arg0) {torch.onnx.approximate = "tanh"} : (!torch.vtensor<[1,1024,64],bf16>) -> !torch.vtensor<[1,1024,64],bf16> 
    return %0 : !torch.vtensor<[1,1024,64],bf16>
  }
}
