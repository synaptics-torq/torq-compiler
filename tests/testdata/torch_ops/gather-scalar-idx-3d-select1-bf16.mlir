module {
  func.func @gather_scalar_idx_select1(%arg0: !torch.vtensor<[16,1,64],bf16>) -> !torch.vtensor<[1,64],bf16> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__idx0> : tensor<si64>} : () -> !torch.vtensor<[],si64>
    %none = torch.constant.none
    %1 = torch.operator "onnx.Gather"(%arg0, %0) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[16,1,64],bf16>, !torch.vtensor<[],si64>) -> !torch.vtensor<[1,64],bf16>
    return %1 : !torch.vtensor<[1,64],bf16>
  }
}

{-#
  dialect_resources: {
    builtin: {
      __idx0: "0x080000000100000000000000"
    }
  }
#-}
