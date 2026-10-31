// Compile-only: single-output-point conv1d (window == whole input,
// dilation 1) with a RUNTIME init.  Nothing pre-folds a runtime init, and
// Conv1DtoMatmulPattern claims only the all-zero-init form, so
// the conv1d->conv2d promotion must not bail on this shape as "matmul
// territory" - before the bail was aligned with the claim (shared
// isConv1dMatmulCandidate predicate) the conv reached tiling
// with no lowering and the compile failed ("linalg.conv_1d_ncw_fcw op
// unsupported in function body").
//
// The numerics are not validated here: the conv2d lowering (as on main)
// does not carry a runtime conv init into the torq_hl op, so the result is
// wrong (synaptics-torq/torq-compiler-dev#2824).
module {
  func.func @main(%arg0: tensor<2x3x1xbf16>, %arg1: tensor<4x3x1xbf16>, %arg2: tensor<2x4x1xbf16>) -> tensor<2x4x1xbf16> {
    %0 = linalg.conv_1d_ncw_fcw {dilations = dense<[1]> : vector<1xi64>, strides = dense<[1]> : vector<1xi64>} ins(%arg0, %arg1 : tensor<2x3x1xbf16>, tensor<4x3x1xbf16>) outs(%arg2 : tensor<2x4x1xbf16>) -> tensor<2x4x1xbf16>
    return %0 : tensor<2x4x1xbf16>
  }
}
