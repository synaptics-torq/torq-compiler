module {
  func.func @Gather_node_upsample_nearest1d_1_standalone(%arg0: !torch.vtensor<[1,64,1280],bf16>, %arg1: !torch.vtensor<[6401],si64>) -> !torch.vtensor<[1,64,6401],bf16> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 19 : si64, torch.onnx_meta.producer_name = "inspect_onnx_node.extract", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.Gather"(%arg0, %arg1) {torch.onnx.axis = 2 : si64} : (!torch.vtensor<[1,64,1280],bf16>, !torch.vtensor<[6401],si64>) -> !torch.vtensor<[1,64,6401],bf16> 
    return %0 : !torch.vtensor<[1,64,6401],bf16>
  }
}

