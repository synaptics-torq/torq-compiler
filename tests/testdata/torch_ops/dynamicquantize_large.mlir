// TORQ_INT_TOL: 2
// The NPU reduces min/max in bf16 and derives the zero point from those values,
// so its ONNX-rounded integer zero point can land one code away from the one the
// f32 reference picks (a bf16 min of -40.0 puts an exact zp of 127.4988 onto the
// 127.5 tie, which rounds to 128 instead of 127). That is a whole-tensor offset of
// one code, on top of the usual one code of bf16 rounding noise. Both sides stay
// self-consistent, since the same integer is published as y_zero_point and embedded
// in the quantized tensor.
// Large DynamicQuantizeLinear whose reduction exceeds LRAM, so the combined
// min+max reduce is tiled into an scf.for and its per-tile partials fold across
// tiles.
module {
  func.func @main(%arg0: !torch.vtensor<[1,1,131072],f32>) -> !torch.vtensor<[1,1,131072],ui8> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64} {
    %0:3 = torch.operator "onnx.DynamicQuantizeLinear"(%arg0) : (!torch.vtensor<[1,1,131072],f32>) -> (!torch.vtensor<[1,1,131072],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>)
    return %0#0 : !torch.vtensor<[1,1,131072],ui8>
  }
}
