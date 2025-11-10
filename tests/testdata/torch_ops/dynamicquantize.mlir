module  {
  func.func @main(%984: !torch.vtensor<[1,1,288],f32>) -> !torch.vtensor<[1,1,288],ui8> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.opset_versions = {ai.onnx.ml = 5 : si64, ai.onnx.preview.training = 1 : si64, ai.onnx.training = 1 : si64, com.microsoft = 1 : si64, com.microsoft.experimental = 1 : si64, com.microsoft.nchwc = 1 : si64, org.pytorch.aten = 1 : si64}, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %985:3 = torch.operator "onnx.DynamicQuantizeLinear"(%984) : (!torch.vtensor<[1,1,288],f32>) -> (!torch.vtensor<[1,1,288],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>)
    return %985#0 : !torch.vtensor<[1,1,288],ui8>
  }
}

{-#
  dialect_resources: {
    builtin: {
      
    }
  }
#-}
