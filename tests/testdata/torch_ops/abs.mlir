module attributes {torch.debug_module_name = "SimpleModule"} {
  func.func @main(%arg0: !torch.vtensor<[2,16],f32>) -> !torch.vtensor<[2,16],f32> {
    %0 = torch.aten.abs %arg0 : !torch.vtensor<[2,16],f32> -> !torch.vtensor<[2,16],f32>
    return %0 : !torch.vtensor<[2,16],f32>
  }
}

