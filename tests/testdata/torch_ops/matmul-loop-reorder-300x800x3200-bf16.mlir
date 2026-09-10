// A [300,800] LHS fits LRAM whole while the [800,3200] weight does not, and M
// still needs tiling, so visiting tiles n-outer streams the weight once instead
// of once per M tile: this shape must take the for(n)for(m) interchange path.
module {
  func.func @main(%a: !torch.vtensor<[300,800],bf16>) -> !torch.vtensor<[300,3200],bf16> attributes {torch.onnx_meta.ir_version = 11 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "2.1.1"} {
    %w = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1.0e-02> : tensor<800x3200xbf16>} : () -> !torch.vtensor<[800,3200],bf16>
    %r = torch.operator "onnx.MatMul"(%a, %w) : (!torch.vtensor<[300,800],bf16>, !torch.vtensor<[800,3200],bf16>) -> !torch.vtensor<[300,3200],bf16>
    return %r : !torch.vtensor<[300,3200],bf16>
  }
}
