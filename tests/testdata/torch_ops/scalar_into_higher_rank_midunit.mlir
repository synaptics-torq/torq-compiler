
module {
  func.func @scalar_into_midunit_result(%s: !torch.vtensor<[],si32>, %t: !torch.vtensor<[2,1,4],si32>) -> !torch.vtensor<[2,1,4],si32> attributes {torch.onnx_meta.opset_version = 19 : si64} {
    %0 = torch.operator "onnx.Sub"(%s, %t) : (!torch.vtensor<[],si32>, !torch.vtensor<[2,1,4],si32>) -> !torch.vtensor<[2,1,4],si32>
    return %0 : !torch.vtensor<[2,1,4],si32>
  }
}
