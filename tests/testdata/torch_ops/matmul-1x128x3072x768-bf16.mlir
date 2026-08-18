// Large-K matmul [1,128,3072] x [3072,768], bf16 (bert FFN second matmul
// shape). The A operand alone ([1,128,3072] bf16 = 768KB) exceeds the LRAM
// budget and no single parallel-dim shrink fits (M-shrink leaves the 4.5MiB
// weight resident, N-shrink leaves A resident), so the early matmul K-split
// rewrites it into an unrolled chain of accumulating K-chunk matmuls before
// TileAndFuse. Legacy behavior crushed M to 1 and N to 86: ~1152 micro NSS
// jobs for this one op, each re-streaming a ~2KB weight sliver.
//
// The K-chunk partial sums accumulate in bf16, adding rounding vs the fp32
// numpy reference (measured: ~12% of elements exceed 1% rel, ~0.8% exceed
// 5%, none exceed 11%):
// TORQ_FP_AVG_TOL: 0.05
// TORQ_FP_MAX_TOL: 0.15
// TORQ_ALLOWED_WRONG: 0.01
module {
  func.func @main(%arg0: !torch.vtensor<[1,128,3072],bf16>) -> !torch.vtensor<[1,128,768],bf16> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "matmul-ksplit-repro", torch.onnx_meta.producer_version = "1.0"} {
    %w2 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<6.250000e-02> : tensor<3072x768xbf16>} : () -> !torch.vtensor<[3072,768],bf16>
    %mm2 = torch.operator "onnx.MatMul"(%arg0, %w2) : (!torch.vtensor<[1,128,3072],bf16>, !torch.vtensor<[3072,768],bf16>) -> !torch.vtensor<[1,128,768],bf16>
    return %mm2 : !torch.vtensor<[1,128,768],bf16>
  }
}
