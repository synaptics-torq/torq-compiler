// TORQ_FP_MAX_TOL: 0.31
// TORQ_FP_AVG_TOL: 0.03
module {
  func.func @main(%arg0: !torch.vtensor<[1,1280,7,7],bf16>) -> !torch.vtensor<[1,1280,1,1],bf16> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "2.9.1+cu128"} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<val_473> : tensor<2xsi64>} : () -> !torch.vtensor<[2],si64> 
    %207 = torch.operator "onnx.ReduceMean"(%arg0, %0) {torch.onnx.keepdims = 1 : si64, torch.onnx.noop_with_empty_axes = 0 : si64} : (!torch.vtensor<[1,1280,7,7],bf16>, !torch.vtensor<[2],si64>) -> !torch.vtensor<[1,1280,1,1],bf16> 
    return %207 : !torch.vtensor<[1,1280,1,1],bf16>
  }
}

{-#
  dialect_resources: {
    builtin: {
      val_473: "0x08000000FFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF"
    }
  }
#-}

