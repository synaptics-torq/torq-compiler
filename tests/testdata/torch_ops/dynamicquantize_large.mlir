// Large DynamicQuantizeLinear whose reduction exceeds LRAM, so the combined
// min+max reduce is tiled into an scf.for and its per-tile partials fold across
// tiles.
module {
  func.func @main(%arg0: !torch.vtensor<[1,1,131072],f32>) -> !torch.vtensor<[1,1,131072],ui8> attributes {torch.onnx_meta.ir_version = 10 : si64, torch.onnx_meta.opset_version = 22 : si64} {
    %0:3 = torch.operator "onnx.DynamicQuantizeLinear"(%arg0) : (!torch.vtensor<[1,1,131072],f32>) -> (!torch.vtensor<[1,1,131072],ui8>, !torch.vtensor<[],f32>, !torch.vtensor<[],ui8>)
    return %0#0 : !torch.vtensor<[1,1,131072],ui8>
  }
}
