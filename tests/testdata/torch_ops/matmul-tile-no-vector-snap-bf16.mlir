// N=126 must be tiled but splits evenly only as 63+63; snapping to the 32-lane
// vector width would give 32+32+32+30 at the same total vector columns -- extra
// tiles and DMA for no compute win, so the tile search must keep 63.
module {
  func.func @main(%a: !torch.vtensor<[64,1600],bf16>) -> !torch.vtensor<[64,126],bf16> attributes {torch.onnx_meta.ir_version = 11 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "2.1.1"} {
    %w = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1.0e-02> : tensor<1600x126xbf16>} : () -> !torch.vtensor<[1600,126],bf16>
    %r = torch.operator "onnx.MatMul"(%a, %w) : (!torch.vtensor<[64,1600],bf16>, !torch.vtensor<[1600,126],bf16>) -> !torch.vtensor<[64,126],bf16>
    return %r : !torch.vtensor<[64,126],bf16>
  }
}
