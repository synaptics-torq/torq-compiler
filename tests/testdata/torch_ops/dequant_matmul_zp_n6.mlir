// Six output columns: the f32 bias row is 24 bytes, which DEBR cannot read in one access.
// TORQ_FP_MAX_TOL: 0.1
// TORQ_ALLOWED_WRONG: 0.1
module {
  func.func @main(%arg0: !torch.vtensor<[32,6],si8>, %arg1: !torch.vtensor<[1,1,32],bf16>) -> !torch.vtensor<[1,1,6],bf16> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense<1.0> : tensor<1x6xbf16>} : () -> !torch.vtensor<[1,6],bf16> 
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<dequant_matmul_zp_n6_zero_points> : tensor<1x6xsi8>} : () -> !torch.vtensor<[1,6],si8> 
    %none = torch.constant.none
    %2 = torch.operator "onnx.DequantizeLinear"(%arg0, %0, %1) {torch.onnx.axis = 0 : si64, torch.onnx.block_size = 32 : si64} : (!torch.vtensor<[32,6],si8>, !torch.vtensor<[1,6],bf16>, !torch.vtensor<[1,6],si8>) -> !torch.vtensor<[32,6],bf16> 
    %3 = torch.operator "onnx.MatMul"(%arg1, %2) : (!torch.vtensor<[1,1,32],bf16>, !torch.vtensor<[32,6],bf16>) -> !torch.vtensor<[1,1,6],bf16> 
    return %3 : !torch.vtensor<[1,1,6],bf16>
  }
}

{-#
  dialect_resources: {
    builtin: {
      dequant_matmul_zp_n6_zero_points: "0x08000000591925501037"
    }
  }
#-}

