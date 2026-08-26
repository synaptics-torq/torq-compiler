module {
  func.func @part1_graph(%arg0: !torch.vtensor<[2],si64>, %arg1: !torch.vtensor<[],si64>) -> !torch.vtensor<[2],si64> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Mul"(%arg0, %arg1) : (!torch.vtensor<[2],si64>, !torch.vtensor<[],si64>) -> !torch.vtensor<[2],si64> 
    return %0 : !torch.vtensor<[2],si64>
  }
}

