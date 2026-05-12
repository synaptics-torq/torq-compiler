module {
  func.func @part128_graph(%arg0: !torch.vtensor<[1,2560,8,8],bf16>) -> !torch.vtensor<[1,2560,1,1],bf16> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.GlobalAveragePool"(%arg0) : (!torch.vtensor<[1,2560,8,8],bf16>) -> !torch.vtensor<[1,2560,1,1],bf16> 
    return %0 : !torch.vtensor<[1,2560,1,1],bf16>
  }
}

