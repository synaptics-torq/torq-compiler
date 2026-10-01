module {
  func.func @matmul_dql_zp_splat(%arg0: !torch.vtensor<[256,1024],si8>, %arg1: !torch.vtensor<[1,1,256],bf16>) -> !torch.vtensor<[1,1,1024],bf16> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Constant"() { torch.onnx.value = dense<0.03125> : tensor<8x1024xbf16> } : () -> !torch.vtensor<[8,1024],bf16>
    %1 = torch.operator "onnx.Constant"() { torch.onnx.value = dense<127> : tensor<8x1024xsi8> } : () -> !torch.vtensor<[8,1024],si8>
    %2 = torch.operator "onnx.DequantizeLinear"(%arg0, %0, %1) {torch.onnx.axis = 0 : si64, torch.onnx.block_size = 32 : si64} : (!torch.vtensor<[256,1024],si8>, !torch.vtensor<[8,1024],bf16>, !torch.vtensor<[8,1024],si8>) -> !torch.vtensor<[256,1024],bf16>
    %3 = torch.operator "onnx.MatMul"(%arg1, %2) : (!torch.vtensor<[1,1,256],bf16>, !torch.vtensor<[256,1024],bf16>) -> !torch.vtensor<[1,1,1024],bf16>
    return %3 : !torch.vtensor<[1,1,1024],bf16>
  }
}
