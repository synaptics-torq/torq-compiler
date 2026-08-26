module {
  func.func @part375_graph(%arg0: !torch.vtensor<[1,2100],bf16>) -> !torch.vtensor<[1,300],si64> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__model.23_Constant_16_output_0_part375_init0> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64> 
    %none = torch.constant.none
    %1:2 = torch.operator "onnx.TopK"(%arg0, %0) {torch.onnx.axis = -1 : si64, torch.onnx.largest = 1 : si64, torch.onnx.sorted = 1 : si64} : (!torch.vtensor<[1,2100],bf16>, !torch.vtensor<[1],si64>) -> (!torch.vtensor<[1,300],bf16>, !torch.vtensor<[1,300],si64>) 
    return %1#1 : !torch.vtensor<[1,300],si64>
  }
}

{-#
  dialect_resources: {
    builtin: {
      __model.23_Constant_16_output_0_part375_init0: "0x080000002C01000000000000"
    }
  }
#-}

