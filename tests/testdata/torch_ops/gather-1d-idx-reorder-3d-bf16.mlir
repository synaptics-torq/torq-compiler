module {
  func.func @gather_1d_idx_reorder_3d(%arg0: !torch.vtensor<[16,1,64],bf16>) -> !torch.vtensor<[16,1,64],bf16> attributes {torch.onnx_meta.ir_version = 7 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<__reverse_indices> : tensor<16xsi64>} : () -> !torch.vtensor<[16],si64>
    %none = torch.constant.none
    %1 = torch.operator "onnx.Gather"(%arg0, %0) {torch.onnx.axis = 0 : si64} : (!torch.vtensor<[16,1,64],bf16>, !torch.vtensor<[16],si64>) -> !torch.vtensor<[16,1,64],bf16>
    return %1 : !torch.vtensor<[16,1,64],bf16>
  }
}

{-#
  dialect_resources: {
    builtin: {
      __reverse_indices: "0x080000000F000000000000000E000000000000000D000000000000000C000000000000000B000000000000000A000000000000000900000000000000080000000000000007000000000000000600000000000000050000000000000004000000000000000300000000000000020000000000000001000000000000000000000000000000"
    }
  }
#-}
