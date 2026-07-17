module {
  func.func @part46_graph(%arg0: !torch.vtensor<[1,512,7,7],si8>) -> !torch.vtensor<[1,512,1,1],si8> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 18 : si64, torch.onnx_meta.producer_name = "onnx.quantize", torch.onnx_meta.producer_version = "0.1.0"} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0> : tensor<si8>} : () -> !torch.vtensor<[],si8> 
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.00784310605> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %2 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<-5> : tensor<si8>} : () -> !torch.vtensor<[],si8> 
    %3 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.00234464626> : tensor<f32>} : () -> !torch.vtensor<[],f32> 
    %none = torch.constant.none
    %4 = torch.operator "onnx.DequantizeLinear"(%arg0, %1, %0) : (!torch.vtensor<[1,512,7,7],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[1,512,7,7],f32> 
    %5 = torch.operator "onnx.GlobalAveragePool"(%4) : (!torch.vtensor<[1,512,7,7],f32>) -> !torch.vtensor<[1,512,1,1],f32> 
    %6 = torch.operator "onnx.QuantizeLinear"(%5, %3, %2) : (!torch.vtensor<[1,512,1,1],f32>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[1,512,1,1],si8> 
    return %6 : !torch.vtensor<[1,512,1,1],si8>
  }
}

