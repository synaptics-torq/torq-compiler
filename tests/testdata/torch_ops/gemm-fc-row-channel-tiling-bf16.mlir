// TORQ_FP_MAX_TOL: 0.05
module {
  func.func @main(%input: !torch.vtensor<[207,1152],bf16>) -> !torch.vtensor<[207,288],bf16> attributes {torch.onnx_meta.ir_version = 11 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "test", torch.onnx_meta.producer_version = "1.0"} {
    %weight = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1.000000e-02> : tensor<1152x288xbf16>} : () -> !torch.vtensor<[1152,288],bf16>
    %bias = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.0> : tensor<288xbf16>} : () -> !torch.vtensor<[288],bf16>
    %output = torch.operator "onnx.Gemm"(%input, %weight, %bias) {torch.onnx.alpha = 1.000000e+00 : f32, torch.onnx.beta = 1.000000e+00 : f32, torch.onnx.transA = 0 : si64, torch.onnx.transB = 0 : si64} : (!torch.vtensor<[207,1152],bf16>, !torch.vtensor<[1152,288],bf16>, !torch.vtensor<[288],bf16>) -> !torch.vtensor<[207,288],bf16>
    return %output : !torch.vtensor<[207,288],bf16>
  }
}

{-#
  dialect_resources: {
    builtin: {
    }
  }
#-}
