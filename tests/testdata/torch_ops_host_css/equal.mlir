module  {
  func.func @main(%270: !torch.vtensor<[1],si64>) -> !torch.vtensor<[1,1,30,1],i1> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.opset_versions = {ai.onnx.ml = 5 : si64, ai.onnx.preview.training = 1 : si64, ai.onnx.training = 1 : si64, com.microsoft = 1 : si64, com.microsoft.experimental = 1 : si64, com.microsoft.nchwc = 1 : si64, org.pytorch.aten = 1 : si64}, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %5 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<time_ids> : tensor<1x1x30x1xsi64>} : () -> !torch.vtensor<[1,1,30,1],si64>
    %337 = torch.operator "onnx.Equal"(%5, %270) : (!torch.vtensor<[1,1,30,1],si64>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[1,1,30,1],i1>
    return %337 : !torch.vtensor<[1,1,30,1],i1>
  }
}

{-#
  dialect_resources: {
    builtin: {
      time_ids: "0x0800000000000000000000000100000000000000020000000000000003000000000000000400000000000000050000000000000006000000000000000700000000000000080000000000000009000000000000000A000000000000000B000000000000000C000000000000000D000000000000000E000000000000000F0000000000000010000000000000001100000000000000120000000000000013000000000000001400000000000000150000000000000016000000000000001700000000000000180000000000000019000000000000001A000000000000001B000000000000001C000000000000001D00000000000000"
    }
  }
#-}
