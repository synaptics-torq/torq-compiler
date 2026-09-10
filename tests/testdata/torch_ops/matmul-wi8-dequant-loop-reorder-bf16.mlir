// int8 weight dequantized by a fused generic: the loop-order byte model must
// price the weight stream at 1 byte and still pick the n-outer order.
// Use an exact power-of-two scale so dequant rounding does not obscure tiling errors.
module {
  func.func @main(%a: !torch.vtensor<[300,800],bf16>, %w: !torch.vtensor<[800,3200],si8>) -> !torch.vtensor<[300,3200],bf16> attributes {torch.onnx_meta.ir_version = 11 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "2.1.1"} {
    %s = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1.5625e-02> : tensor<bf16>} : () -> !torch.vtensor<[],bf16>
    %z = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si8>} : () -> !torch.vtensor<[],si8>
    %dq = torch.operator "onnx.DequantizeLinear"(%w, %s, %z) : (!torch.vtensor<[800,3200],si8>, !torch.vtensor<[],bf16>, !torch.vtensor<[],si8>) -> !torch.vtensor<[800,3200],bf16>
    %r = torch.operator "onnx.MatMul"(%a, %dq) : (!torch.vtensor<[300,800],bf16>, !torch.vtensor<[800,3200],bf16>) -> !torch.vtensor<[300,3200],bf16>
    return %r : !torch.vtensor<[300,3200],bf16>
  }
}
