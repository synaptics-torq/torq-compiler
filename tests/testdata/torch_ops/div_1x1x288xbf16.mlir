module {
  func.func @main(%arg0: !torch.vtensor<[1,1,288],bf16>, %arg1: !torch.vtensor<[1,1,1],bf16>) -> !torch.vtensor<[1,1,288],bf16> attributes {torch.onnx_meta.ir_version = 12 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Div"(%arg0, %arg1) : (!torch.vtensor<[1,1,288],bf16>, !torch.vtensor<[1,1,1],bf16>) -> !torch.vtensor<[1,1,288],bf16> 
    return %0 : !torch.vtensor<[1,1,288],bf16>
  }
}

