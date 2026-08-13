
module {
  func.func @add_scalar_into_unit_dim(%s: !torch.vtensor<[],si32>, %t: !torch.vtensor<[1,256],si32>) -> !torch.vtensor<[1,256],si32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
    %0 = torch.operator "onnx.Add"(%s, %t) : (!torch.vtensor<[],si32>, !torch.vtensor<[1,256],si32>) -> !torch.vtensor<[1,256],si32>
    return %0 : !torch.vtensor<[1,256],si32>
  }
}
