module  {
  func.func @main(%226: !torch.vtensor<[1,64,1,1],bf16>) -> !torch.vtensor<[1,1,1,64],bf16> attributes {torch.onnx_meta.ir_version = 11 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "2.1.1"} {
    %227 = torch.operator "onnx.Transpose"(%226) {torch.onnx.perm = [0 : si64, 3 : si64, 2 : si64, 1 : si64]} : (!torch.vtensor<[1,64,1,1],bf16>) -> !torch.vtensor<[1,1,1,64],bf16>
    return %227 : !torch.vtensor<[1,1,1,64],bf16>
  }
}

{-#
  dialect_resources: {
    builtin: {
      
    }
  }
#-}
