module {
  func.func @main(%arg1: !torch.vtensor<[3,3,1],f32>, %arg2: !torch.vtensor<[3,3,1],f32>) -> (!torch.vtensor<[3,3,2],f32>) attributes {torch.onnx_meta.ir_version = 6 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "2.1.1"} {
    %1 = torch.operator "onnx.Cast"(%arg1) {torch.onnx.to = 16 : si64} : (!torch.vtensor<[3,3,1],f32>) -> !torch.vtensor<[3,3,1],si8> 
    %2 = torch.operator "onnx.Cast"(%arg2) {torch.onnx.to = 16 : si64} : (!torch.vtensor<[3,3,1],f32>) -> !torch.vtensor<[3,3,1],si8> 
    %3 = torch.operator "onnx.Concat"(%1, %2) {torch.onnx.axis = 2 : si64} : (!torch.vtensor<[3,3,1],si8>, !torch.vtensor<[3,3,1],si8>) -> !torch.vtensor<[3,3,2],si8> 
    %4 = torch.operator "onnx.Cast"(%3) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[3,3,2],si8>) -> !torch.vtensor<[3,3,2],f32> 
    return %4 : !torch.vtensor<[3,3,2],f32>
  }
}

{-#
  dialect_resources: {
    builtin: {
    }
  }
#-}
