// TORQ_USE_ABS_TOL_GATE: 1
// TORQ_FP_ABS_TOL_FRAC: 0.015
module {
  func.func @matmulinteger_depthwise_conv(%arg0: !torch.vtensor<[4,1,3],f32>) -> !torch.vtensor<[4,1,1],f32> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 21 : si64, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %0 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<W> : tensor<4x3x1xsi8>} : () -> !torch.vtensor<[4,3,1],si8>
    %1 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<w_zp> : tensor<1xsi8>} : () -> !torch.vtensor<[1],si8>
    %2 = torch.operator "onnx.Constant"() {torch.onnx.value = dense_resource<w_scale> : tensor<1xf32>} : () -> !torch.vtensor<[1],f32>
    %3:3 = torch.operator "onnx.DynamicQuantizeLinear"(%arg0) : (!torch.vtensor<[4,1,3],f32>) -> (!torch.vtensor<[4,1,3],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>)
    %4 = torch.operator "onnx.MatMulInteger"(%3#0, %0, %3#2, %1) : (!torch.vtensor<[4,1,3],ui8>, !torch.vtensor<[4,3,1],si8>, !torch.vtensor<[],ui8>, !torch.vtensor<[1],si8>) -> !torch.vtensor<[4,1,1],si32>
    %5 = torch.operator "onnx.Cast"(%4) {torch.onnx.to = 1 : si64} : (!torch.vtensor<[4,1,1],si32>) -> !torch.vtensor<[4,1,1],f32>
    %6 = torch.operator "onnx.Mul"(%3#1, %2) : (!torch.vtensor<[],f32>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[1],f32>
    %7 = torch.operator "onnx.Mul"(%5, %6) : (!torch.vtensor<[4,1,1],f32>, !torch.vtensor<[1],f32>) -> !torch.vtensor<[4,1,1],f32>
    return %7 : !torch.vtensor<[4,1,1],f32>
  }
}

{-#
  dialect_resources: {
    builtin: {
      W: "0x080000000102030405060708090A0B0C",
      w_zp: "0x0800000000",
      w_scale: "0x080000000000003C"
    }
  }
#-}
