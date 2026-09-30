// TORQ_FP_MAX_TOL: 0.091
module {
  func.func @main(%arg0: !torch.vtensor<[1,1,359712],bf16>) -> !torch.vtensor<[1,1,1],bf16> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<[-1]> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %none = torch.constant.none
    %2 = torch.operator "onnx.ReduceMean"(%arg0, %0) : (!torch.vtensor<[1,1,359712],bf16>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[1,1,1],bf16> 
    return %2 : !torch.vtensor<[1,1,1],bf16>
  }
}

