// TORQ_INT_TOL: 2
// Regression for the derive_quant_params reciprocal LUT: inputs in [-1, 1) put the bf16
// range at exactly 2.0 (mantissa 0), where the kernel used to compute 1/range 4x too
// small, so y and the zero point came out at a quarter of the right codes (error up to
// ~190) while y_scale stayed correct. Ranges with mantissa 127 (just below a power of
// two) hit the same entry. The tolerance is the one of dynamicquantize.mlir: the zero
// point can land one code away from the f32 reference, plus one code of bf16 rounding.
module {
  func.func @main(%arg0: !torch.vtensor<[1,1,1024],f32>) -> !torch.vtensor<[1,1,1024],ui8> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64} {
    %0:3 = torch.operator "onnx.DynamicQuantizeLinear"(%arg0) : (!torch.vtensor<[1,1,1024],f32>) -> (!torch.vtensor<[1,1,1024],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>)
    return %0#0 : !torch.vtensor<[1,1,1024],ui8>
  }
}
