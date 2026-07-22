module {
  func.func @part1_graph(%arg0: !torch.vtensor<[1,32,112,112],si8>) -> !torch.vtensor<[1,32,112,112],si8> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 17 : si64, torch.onnx_meta.opset_versions = {com.microsoft = 1 : si64}, torch.onnx_meta.producer_name = "onnx.quantize", torch.onnx_meta.producer_version = "0.1.0"} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<-128> : tensor<si8>} : () -> !torch.vtensor<[],si8> 
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.00286689634> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %2 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si8>} : () -> !torch.vtensor<[],si8> 
    %3 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.00784313586> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %none = torch.constant.none
    %4 = torch.operator "onnx.QLinearSigmoid"(%arg0, %3, %2, %1, %0) : (!torch.vtensor<[1,32,112,112],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[1,32,112,112],si8> 
    return %4 : !torch.vtensor<[1,32,112,112],si8>
  }
}

