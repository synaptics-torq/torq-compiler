// Depthwise Conv1D (group == channels) whose per-channel bias is materialized
// as a compile-time-const `linalg.reduce` inside a tiled scf.for/scf.forall
// nest. The channel count (464) forces the conv to tile into a two-level loop
// nest, so the const-outline pass (CompileTimeConstOutlinePass) clones a
// def-chain that reads the loop's shared-out destination. Regression guard for
// that outlining path emitting an invalid extract_slice (wrong-typed source /
// mismatched offset count) and aborting the compile. Extracted from the tsuki
// part_a_best model; see issue #1954.
module {
  func.func @depthwise_conv1d_c464_k5(%arg0: !torch.vtensor<[1,464,320],bf16>) -> !torch.vtensor<[1,464,320],bf16> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 19 : si64} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<3.000000e-01> : tensor<464x1x5xbf16>} : () -> !torch.vtensor<[464,1,5],bf16>
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1.000000e-01> : tensor<464xbf16>} : () -> !torch.vtensor<[464],bf16>
    %2 = torch.operator "onnx.Conv"(%arg0, %0, %1) {torch.onnx.dilations = [1 : si64], torch.onnx.group = 464 : si64, torch.onnx.kernel_shape = [5 : si64], torch.onnx.pads = [2 : si64, 2 : si64], torch.onnx.strides = [1 : si64]} : (!torch.vtensor<[1,464,320],bf16>, !torch.vtensor<[464,1,5],bf16>, !torch.vtensor<[464],bf16>) -> !torch.vtensor<[1,464,320],bf16>
    return %2 : !torch.vtensor<[1,464,320],bf16>
  }
}
