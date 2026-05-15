module {
  func.func @part24_graph(%arg0: !torch.vtensor<[1,192,35,35],bf16>) -> !torch.vtensor<[1,192,35,35],bf16> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %none = torch.constant.none
    %0 = torch.operator "onnx.AveragePool"(%arg0) {torch.onnx.ceil_mode = 0 : si64, torch.onnx.count_include_pad = 1 : si64, torch.onnx.kernel_shape = [3 : si64, 3 : si64], torch.onnx.pads = [1 : si64, 1 : si64, 1 : si64, 1 : si64], torch.onnx.strides = [1 : si64, 1 : si64]} : (!torch.vtensor<[1,192,35,35],bf16>) -> !torch.vtensor<[1,192,35,35],bf16> 
    return %0 : !torch.vtensor<[1,192,35,35],bf16>
  }
}

