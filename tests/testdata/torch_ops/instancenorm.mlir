module  {
  func.func @main(%232: !torch.vtensor<[1,1,359712],bf16>) -> !torch.vtensor<[1,1,359712],bf16> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.opset_versions = {ai.onnx.ml = 5 : si64, ai.onnx.preview.training = 1 : si64, ai.onnx.training = 1 : si64, com.microsoft = 1 : si64, com.microsoft.experimental = 1 : si64, com.microsoft.nchwc = 1 : si64, org.pytorch.aten = 1 : si64}, torch.onnx_meta.producer_name = "onnx.quantize", torch.onnx_meta.producer_version = "0.1.0"} {
    %3 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__groupnorm_Constant_1_output_0_bf16> : tensor<1xbf16>} : () -> !torch.vtensor<[1],bf16>
    %4 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__groupnorm_Constant_2_output_0_bf16> : tensor<1xbf16>} : () -> !torch.vtensor<[1],bf16>
    %233 = torch.operator "onnx.InstanceNormalization"(%232, %3, %4) {torch.onnx.epsilon = 9.99999974E-6 : f32} : (!torch.vtensor<[1,1,359712],bf16>, !torch.vtensor<[1],bf16>, !torch.vtensor<[1],bf16>) -> !torch.vtensor<[1,1,359712],bf16>
    return %233 : !torch.vtensor<[1,1,359712],bf16>
  }
}

{-#
  dialect_resources: {
    builtin: {
      __groupnorm_Constant_1_output_0_bf16: "0x08000000803F",
      __groupnorm_Constant_2_output_0_bf16: "0x080000000000"
    }
  }
#-}
