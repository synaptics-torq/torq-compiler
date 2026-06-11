module {
  func.func @matmul_dql_test(%arg0: !torch.vtensor<[640,205],si8>, %arg1: !torch.vtensor<[1,1,640],bf16>) -> !torch.vtensor<[1,1,205],bf16> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.opset_versions = {ai.onnx.ml = 5 : si64, ai.onnx.preview.training = 1 : si64, ai.onnx.training = 1 : si64, com.microsoft = 1 : si64, com.microsoft.experimental = 1 : si64, com.microsoft.nchwc = 1 : si64, org.pytorch.aten = 1 : si64}, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Constant"() { torch.onnx.value = dense<1.0> : tensor<20x205xbf16> } : () -> !torch.vtensor<[20,205],bf16>
    %1 = torch.operator "onnx.Constant"() { torch.onnx.value = dense<30> : tensor<20x205xsi8>  } : () -> !torch.vtensor<[20,205],si8>
    %2 = torch.operator "onnx.DequantizeLinear"(%arg0, %0, %1) {torch.onnx.axis = 0 : si64, torch.onnx.block_size = 32 : si64} : (!torch.vtensor<[640,205],si8>, !torch.vtensor<[20,205],bf16>, !torch.vtensor<[20,205],si8>) -> !torch.vtensor<[640,205],bf16>
    %4 = torch.operator "onnx.MatMul"(%arg1, %2) : (!torch.vtensor<[1,1,640],bf16>, !torch.vtensor<[640,205],bf16>) -> !torch.vtensor<[1,1,205],bf16> 
    return %4 : !torch.vtensor<[1,1,205],bf16>
  }
}
