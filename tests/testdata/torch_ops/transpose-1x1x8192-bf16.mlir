module  {
  func.func @main(%205: !torch.vtensor<[1,1,8192],bf16>) -> !torch.vtensor<[1,8192,1],bf16> attributes {torch.onnx_meta.ir_version = 11 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "2.1.1"} {
    %206 = torch.operator "onnx.Transpose"(%205) {torch.onnx.perm = [1 : si64, 2 : si64, 0 : si64]} : (!torch.vtensor<[1,1,8192],bf16>) -> !torch.vtensor<[1,8192,1],bf16>
    return %206 : !torch.vtensor<[1,8192,1],bf16>
  }
}

{-#
  dialect_resources: {
    builtin: {
      
    }
  }
#-}
