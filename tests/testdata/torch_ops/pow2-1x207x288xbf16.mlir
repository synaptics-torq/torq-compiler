module {
  func.func @main(%arg0: !torch.vtensor<[1,207,288],bf16>) -> !torch.vtensor<[1,207,288],bf16> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<2.000000e+00> : tensor<bf16>} : () -> !torch.vtensor<[],bf16> 
    %1 = torch.operator "onnx.Pow"(%arg0, %0) : (!torch.vtensor<[1,207,288],bf16>, !torch.vtensor<[],bf16>) -> !torch.vtensor<[1,207,288],bf16> 
    return %1 : !torch.vtensor<[1,207,288],bf16>
  }
}
