module {
  func.func @part0_graph() -> !torch.vtensor<[2],si64> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__Constant_3_output_0_part0_init0> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %none = torch.constant.none
    %1 = torch.operator "onnx.ConstantOfShape"(%0) {torch.onnx.value = dense_resource<_> : tensor<1xsi64>} : (!torch.vtensor<[1],si64>) -> !torch.vtensor<[2],si64> 
    return %1 : !torch.vtensor<[2],si64>
  }
}

{-#
  dialect_resources: {
    builtin: {
      __Constant_3_output_0_part0_init0: "0x080000000200000000000000",
      _: "0x080000000100000000000000"
    }
  }
#-}

