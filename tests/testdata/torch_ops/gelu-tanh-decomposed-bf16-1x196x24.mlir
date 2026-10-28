module {
  func.func @main(%arg0: !torch.vtensor<[1,196,24],bf16>) -> !torch.vtensor<[1,196,24],bf16> attributes {torch.onnx_meta.ir_version = 8 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.producer_name = "pytorch", torch.onnx_meta.producer_version = "2.11.0"} {
    %one = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1.000000e+00> : tensor<bf16>} : () -> !torch.vtensor<[],bf16>
    %half = torch.operator "onnx.Constant"() {torch.onnx.value = dense<5.000000e-01> : tensor<bf16>} : () -> !torch.vtensor<[],bf16>
    %three = torch.operator "onnx.Constant"() {torch.onnx.value = dense<3.000000e+00> : tensor<bf16>} : () -> !torch.vtensor<[],bf16>
    %coeff = torch.operator "onnx.Constant"() {torch.onnx.value = dense<4.467770e-02> : tensor<bf16>} : () -> !torch.vtensor<[],bf16>
    %sqrt2pi = torch.operator "onnx.Constant"() {torch.onnx.value = dense<7.968750e-01> : tensor<bf16>} : () -> !torch.vtensor<[],bf16>
    %0 = torch.operator "onnx.Mul"(%arg0, %half) : (!torch.vtensor<[1,196,24],bf16>, !torch.vtensor<[],bf16>) -> !torch.vtensor<[1,196,24],bf16>
    %1 = torch.operator "onnx.Pow"(%arg0, %three) : (!torch.vtensor<[1,196,24],bf16>, !torch.vtensor<[],bf16>) -> !torch.vtensor<[1,196,24],bf16>
    %2 = torch.operator "onnx.Mul"(%1, %coeff) : (!torch.vtensor<[1,196,24],bf16>, !torch.vtensor<[],bf16>) -> !torch.vtensor<[1,196,24],bf16>
    %3 = torch.operator "onnx.Add"(%arg0, %2) : (!torch.vtensor<[1,196,24],bf16>, !torch.vtensor<[1,196,24],bf16>) -> !torch.vtensor<[1,196,24],bf16>
    %4 = torch.operator "onnx.Mul"(%3, %sqrt2pi) : (!torch.vtensor<[1,196,24],bf16>, !torch.vtensor<[],bf16>) -> !torch.vtensor<[1,196,24],bf16>
    %5 = torch.operator "onnx.Tanh"(%4) : (!torch.vtensor<[1,196,24],bf16>) -> !torch.vtensor<[1,196,24],bf16>
    %6 = torch.operator "onnx.Add"(%5, %one) : (!torch.vtensor<[1,196,24],bf16>, !torch.vtensor<[],bf16>) -> !torch.vtensor<[1,196,24],bf16>
    %7 = torch.operator "onnx.Mul"(%0, %6) : (!torch.vtensor<[1,196,24],bf16>, !torch.vtensor<[1,196,24],bf16>) -> !torch.vtensor<[1,196,24],bf16>
    return %7 : !torch.vtensor<[1,196,24],bf16>
  }
}
