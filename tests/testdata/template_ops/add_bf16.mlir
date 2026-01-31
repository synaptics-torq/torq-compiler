module {
  func.func @main(%arg0: !torch.vtensor<{shape_4_bf16},bf16>, %arg1: !torch.vtensor<{shape_4_bf16},bf16>) -> !torch.vtensor<{shape_4_bf16},bf16> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Add"(%arg0, %arg1) : (!torch.vtensor<{shape_4_bf16},bf16>, !torch.vtensor<{shape_4_bf16},bf16>) -> !torch.vtensor<{shape_4_bf16},bf16> 
    return %0 : !torch.vtensor<{shape_4_bf16},bf16>
  }
}

