module  {
  func.func @main(%202: !torch.vtensor<[1,512,16,1],bf16>) -> !torch.vtensor<[1,1,512,16],bf16> attributes {torch.onnx_meta.ir_version = 11 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "2.1.1"} {
    %203 = torch.operator "onnx.Transpose"(%202) {torch.onnx.perm = [3 : si64, 0 : si64, 1 : si64, 2 : si64]} : (!torch.vtensor<[1,512,16,1],bf16>) -> !torch.vtensor<[1,1,512,16],bf16>
    return %203 : !torch.vtensor<[1,1,512,16],bf16>
  }
}

{-#
  dialect_resources: {
    builtin: {
      
    }
  }
#-}
