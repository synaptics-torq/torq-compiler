module {
  func.func @deq_min(%arg0: !torch.vtensor<[1,64,64],si8>) -> !torch.vtensor<[1,64,64],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 19 : si64} {
    %scale = torch.operator "onnx.Constant"() {torch.onnx.value = dense<0.00784313772> : tensor<f32>} : () -> !torch.vtensor<[],f32>
    %zp = torch.operator "onnx.Constant"() {torch.onnx.value = dense<-7> : tensor<si8>} : () -> !torch.vtensor<[],si8>
    %dq = torch.operator "onnx.DequantizeLinear"(%arg0, %scale, %zp) : (!torch.vtensor<[1,64,64],si8>, !torch.vtensor<[],f32>, !torch.vtensor<[],si8>) -> !torch.vtensor<[1,64,64],f32>
    return %dq : !torch.vtensor<[1,64,64],f32>
  }
}
