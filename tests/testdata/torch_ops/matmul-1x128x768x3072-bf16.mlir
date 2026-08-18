// Large-K*N matmul [1,128,768] x [768,3072], bf16 (bert FFN first matmul
// shape). The weight alone (768x3072x2B = 4.5MiB) exceeds the LRAM budget,
// but shrinking the parallel N dim fits, so TileAndFuse handles it — the
// reduction-aware shrink order tiles N instead of crushing M to 1 (legacy
// order produced [M=7,N=308] tiles: 190 micro tiles, ~18x weight
// re-streaming, and a compile-time fit-check storm).
module {
  func.func @main(%arg0: !torch.vtensor<[1,128,768],bf16>) -> !torch.vtensor<[1,128,3072],bf16> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "matmul-fit-repro", torch.onnx_meta.producer_version = "1.0"} {
    %w1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<6.250000e-02> : tensor<768x3072xbf16>} : () -> !torch.vtensor<[768,3072],bf16>
    %mm1 = torch.operator "onnx.MatMul"(%arg0, %w1) : (!torch.vtensor<[1,128,768],bf16>, !torch.vtensor<[768,3072],bf16>) -> !torch.vtensor<[1,128,3072],bf16>
    return %mm1 : !torch.vtensor<[1,128,3072],bf16>
  }
}
