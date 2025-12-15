module {
  func.func @main(%arg0: !torch.vtensor<[1,256,128,1],f32>, %arg1: !torch.vtensor<[1,256,128,1],f32>) -> (!torch.vtensor<[1,256,128,2],f32>) attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "2.1.1"} {
    %1 = torch.operator "onnx.Cast"(%arg0) {torch.onnx.to = 16 : si64} : (!torch.vtensor<[1,256,128,1],f32>) -> !torch.vtensor<[1,256,128,1],bf16> 
    %2 = torch.operator "onnx.Cast"(%arg1) {torch.onnx.to = 16 : si64} : (!torch.vtensor<[1,256,128,1],f32>) -> !torch.vtensor<[1,256,128,1],bf16> 
    %3 = torch.operator "onnx.Concat"(%1, %2) {torch.onnx.axis = 3 : si64} : (!torch.vtensor<[1,256,128,1],bf16>, !torch.vtensor<[1,256,128,1],bf16>) -> !torch.vtensor<[1,256,128,2],bf16> 
    %4 = torch.operator "onnx.Cast"(%3) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[1,256,128,2],bf16>) -> !torch.vtensor<[1,256,128,2],f32> 
    return %4 : !torch.vtensor<[1,256,128,2],f32>
  }
}

{-#
  dialect_resources: {
    builtin: {
    }
  }
#-}

