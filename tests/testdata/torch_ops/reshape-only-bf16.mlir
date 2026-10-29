module {
  func.func @main(%arg0: !torch.vtensor<[1,256,20,20],bf16>) -> !torch.vtensor<[1,256,400],bf16> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 19 : si64} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<shape> : tensor<3xsi64>} : () -> !torch.vtensor<[3],si64>
    %1 = torch.operator "onnx.Reshape"(%arg0, %0) : (!torch.vtensor<[1,256,20,20],bf16>, !torch.vtensor<[3],si64>) -> !torch.vtensor<[1,256,400],bf16>
    return %1 : !torch.vtensor<[1,256,400],bf16>
  }
}
{-#
  dialect_resources: {
    builtin: {
      shape: "0x08000000010000000000000000010000000000009001000000000000"
    }
  }
#-}
