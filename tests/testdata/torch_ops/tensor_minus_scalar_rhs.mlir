
module {
  func.func @tensor_minus_scalar_rhs(%t: !torch.vtensor<[1,256],si32>, %s: !torch.vtensor<[],si32>) -> !torch.vtensor<[1,256],si32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
    %0 = torch.operator "onnx.Sub"(%t, %s) : (!torch.vtensor<[1,256],si32>, !torch.vtensor<[],si32>) -> !torch.vtensor<[1,256],si32>
    return %0 : !torch.vtensor<[1,256],si32>
  }
}
