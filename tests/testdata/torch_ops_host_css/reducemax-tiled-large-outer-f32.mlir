// Regression test for synaptics-torq/torq-compiler-dev#2632: an f32 reduction large
// enough to be tiled sends its partial-max generic to the host, which reads a
// column of the partial buffer (memref<4x162x1xf32, strided<[648, 4, 4]>>). The
// runtime took the innermost stride (16 bytes) as the element size, saw the column
// as contiguous and copied it flat, so most rows of the result were wrong.
func.func @main(%x: !torch.vtensor<[4,162,1,512],f32>) -> !torch.vtensor<[4,162,1,1],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "2.1.1"} {
  %axis = torch.operator "onnx.Constant"() {torch.onnx.value = dense<3> : tensor<1xsi64>} : () -> !torch.vtensor<[1],si64>
  %r = torch.operator "onnx.ReduceMax"(%x, %axis) {torch.onnx.keepdims = 1 : si64, torch.onnx.noop_with_empty_axes = 0 : si64} : (!torch.vtensor<[4,162,1,512],f32>, !torch.vtensor<[1],si64>) -> !torch.vtensor<[4,162,1,1],f32>
  return %r : !torch.vtensor<[4,162,1,1],f32>
}
