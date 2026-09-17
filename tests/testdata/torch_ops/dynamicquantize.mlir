// TORQ_INT_TOL: 2
// The NPU reduces min/max in bf16 and derives the zero point from those values,
// so its ONNX-rounded integer zero point can land one code away from the one the
// f32 reference picks (a bf16 min of -40.0 puts an exact zp of 127.4988 onto the
// 127.5 tie, which rounds to 128 instead of 127). That is a whole-tensor offset of
// one code, on top of the usual one code of bf16 rounding noise. Both sides stay
// self-consistent, since the same integer is published as y_zero_point and embedded
// in the quantized tensor.
module  {
  func.func @main(%984: !torch.vtensor<[1,1,288],f32>) -> !torch.vtensor<[1,1,288],ui8> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64, torch.onnx_meta.opset_versions = {ai.onnx.ml = 5 : si64, ai.onnx.preview.training = 1 : si64, ai.onnx.training = 1 : si64, com.microsoft = 1 : si64, com.microsoft.experimental = 1 : si64, com.microsoft.nchwc = 1 : si64, org.pytorch.aten = 1 : si64}, torch.onnx_meta.producer_name = "", torch.onnx_meta.producer_version = ""} {
    %985:3 = torch.operator "onnx.DynamicQuantizeLinear"(%984) : (!torch.vtensor<[1,1,288],f32>) -> (!torch.vtensor<[1,1,288],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>)
    return %985#0 : !torch.vtensor<[1,1,288],ui8>
  }
}

{-#
  dialect_resources: {
    builtin: {
      
    }
  }
#-}
