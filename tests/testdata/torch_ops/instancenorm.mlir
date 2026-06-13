module {
  func.func @main(%arg0: !torch.vtensor<[1,1,359712],bf16>) -> !torch.vtensor<[1,1,359712],bf16> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__groupnorm_Constant_1_output_0_bf16_part16_init0> : tensor<1xbf16>} : () -> !torch.vtensor<[1],bf16> 
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__groupnorm_Constant_2_output_0_bf16_part16_init1> : tensor<1xbf16>} : () -> !torch.vtensor<[1],bf16> 
    %none = torch.constant.none
    %2 = torch.operator "onnx.InstanceNormalization"(%arg0, %0, %1) {torch.onnx.epsilon = 9.99999974E-6 : f32} : (!torch.vtensor<[1,1,359712],bf16>, !torch.vtensor<[1],bf16>, !torch.vtensor<[1],bf16>) -> !torch.vtensor<[1,1,359712],bf16> 
    return %2 : !torch.vtensor<[1,1,359712],bf16>
  }
}

{-#
  dialect_resources: {
    builtin: {
      __groupnorm_Constant_1_output_0_bf16_part16_init0: "0x08000000803F",
      __groupnorm_Constant_2_output_0_bf16_part16_init1: "0x080000000000"
    }
  }
#-}

