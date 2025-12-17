module  {
  func.func @main(%7861: !torch.vtensor<[1,1,1,1],bf16>) -> !torch.vtensor<[1,1,1,1],bf16> attributes {torch.onnx_meta.ir_version = 11 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "2.1.1"} {
    %7862 = torch.operator "onnx.Sqrt"(%7861) : (!torch.vtensor<[1,1,1,1],bf16>) -> !torch.vtensor<[1,1,1,1],bf16>
    return %7862 : !torch.vtensor<[1,1,1,1],bf16>
  }
}

{-#
  dialect_resources: {
    builtin: {
      
    }
  }
#-}