// Zero-bias GEMM with rows >> channels: any M tile has rows > 8 output channels, so a
// bias declared batch-scaled but sized per-channel indexes out of bounds when lowered.
module {
  func.func @main(%a: !torch.vtensor<[2070,1152],bf16>) -> !torch.vtensor<[2070,8],bf16> attributes {torch.onnx_meta.ir_version = 11 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "2.1.1"} {
    %w = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1.0e-02> : tensor<1152x8xbf16>} : () -> !torch.vtensor<[1152,8],bf16>
    %r = torch.operator "onnx.MatMul"(%a, %w) : (!torch.vtensor<[2070,1152],bf16>, !torch.vtensor<[1152,8],bf16>) -> !torch.vtensor<[2070,8],bf16>
    return %r : !torch.vtensor<[2070,8],bf16>
  }
}
