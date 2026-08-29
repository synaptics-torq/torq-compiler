module {
  func.func @main(%arg0: !torch.vtensor<[1,4,1,256],bf16>) -> !torch.vtensor<[1,4,1,256],bf16> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 20 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<default> : tensor<1xbf16>} : () -> !torch.vtensor<[1],bf16>
    %none = torch.constant.none
    %1 = torch.operator "onnx.Softmax"(%arg0) : (!torch.vtensor<[1,4,1,256],bf16>) -> !torch.vtensor<[1,4,1,256],bf16>
    %2 = torch.operator "onnx.IsNaN"(%1) : (!torch.vtensor<[1,4,1,256],bf16>) -> !torch.vtensor<[1,4,1,256],i1>
    %3 = torch.operator "onnx.Where"(%2, %0, %1) : (!torch.vtensor<[1,4,1,256],i1>, !torch.vtensor<[1],bf16>, !torch.vtensor<[1,4,1,256],bf16>) -> !torch.vtensor<[1,4,1,256],bf16>
    return %3 : !torch.vtensor<[1,4,1,256],bf16>
  }
}

{-#
  dialect_resources: {
    builtin: {
      default: "0x080000000000"
    }
  }
#-}
