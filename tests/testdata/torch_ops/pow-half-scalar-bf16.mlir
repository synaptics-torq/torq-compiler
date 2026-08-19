module {
  func.func @part83_graph(%arg0: !torch.vtensor<[1,1],bf16>) -> !torch.vtensor<[1,1],bf16> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__model_Constant_34_output_0_folded_bf16_part83_init0> : tensor<bf16>} : () -> !torch.vtensor<[],bf16> 
    %none = torch.constant.none
    %1 = torch.operator "onnx.Pow"(%arg0, %0) : (!torch.vtensor<[1,1],bf16>, !torch.vtensor<[],bf16>) -> !torch.vtensor<[1,1],bf16> 
    return %1 : !torch.vtensor<[1,1],bf16>
  }
}

{-#
  dialect_resources: {
    builtin: {
      __model_Constant_34_output_0_folded_bf16_part83_init0: "0x08000000003F"
    }
  }
#-}

