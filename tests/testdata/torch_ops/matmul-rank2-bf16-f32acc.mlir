// bf16 MatMul that lowers to linalg.matmul; torch-mlir accumulates it in f32.
module {
  func.func @main(%arg0: !torch.vtensor<[4,8],bf16>, %arg1: !torch.vtensor<[8,16],bf16>) -> !torch.vtensor<[4,16],bf16> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 19 : si64} {
    %0 = torch.operator "onnx.MatMul"(%arg0, %arg1) : (!torch.vtensor<[4,8],bf16>, !torch.vtensor<[8,16],bf16>) -> !torch.vtensor<[4,16],bf16>
    return %0 : !torch.vtensor<[4,16],bf16>
  }
}
