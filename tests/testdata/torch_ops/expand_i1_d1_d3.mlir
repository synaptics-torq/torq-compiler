module {
  func.func @main(%arg0: !torch.vtensor<[1,1,30,1],i1>) -> !torch.vtensor<[1,8,30,36],i1> attributes {torch.onnx_meta.ir_version = 12 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<_present.0.decoder.key_mask_eq_bcast_shape> : tensor<4xsi64>} : () -> !torch.vtensor<[4],si64> 
    %none = torch.constant.none
    %1 = torch.operator "onnx.Expand"(%arg0, %0) : (!torch.vtensor<[1,1,30,1],i1>, !torch.vtensor<[4],si64>) -> !torch.vtensor<[1,8,30,36],i1> 
    return %1 : !torch.vtensor<[1,8,30,36],i1>
  }
}

{-#
  dialect_resources: {
    builtin: {
      _present.0.decoder.key_mask_eq_bcast_shape: "0x08000000010000000000000008000000000000001E000000000000002400000000000000"
    }
  }
#-}

