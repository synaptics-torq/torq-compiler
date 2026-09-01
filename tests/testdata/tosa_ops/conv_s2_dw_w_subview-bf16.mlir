// TORQ_FP_MAX_TOL: 0.05
// TORQ_USE_ABS_TOL_GATE: 1
// TORQ_FP_ABS_TOL_FRAC: 0.001
// Regression: a stride-2 conv2d fused with a following 3x3 stride-1 depthwise conv
// must compile when tile-and-fuse splits the spatial W dimension. The W-interior
// depthwise tiles consume a W-subview of the producer tile (row pitch != row width),
// which the depthwise HW lowering used to reject with
//   Kernel.cpp: fuse(LData&, int): Assertion `fused >= count` (SIGABRT).
// Mirrors the PP-OCRv6 rec_w1280 backbone block conv[1,24,12,1280]->dw[1,48,6,640].
// Weights are bf16-exact splat constants: the failure is a compile-time layout
// property (W-subview strides), so weight values are irrelevant, and splats keep
// this file ~2 KB instead of 162 KB.
module attributes {tf_saved_model.semantics} {
  func.func @main(%arg0: tensor<1x12x1280x24xbf16> {ml_program.identifier = "serving_default_input_0:0", tf_saved_model.index_path = ["input_0"]}) -> (tensor<1x6x640x48xbf16> {ml_program.identifier = "StatefulPartitionedCall_1:0", tf_saved_model.index_path = ["output_0"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %conv_w = "tosa.const"() <{values = dense<6.250000e-02> : tensor<48x3x3x24xbf16>}> : () -> tensor<48x3x3x24xbf16>
    %conv_b = "tosa.const"() <{values = dense<1.250000e-01> : tensor<48xbf16>}> : () -> tensor<48xbf16>
    %dw_w = "tosa.const"() <{values = dense<-3.125000e-02> : tensor<3x3x48x1xbf16>}> : () -> tensor<3x3x48x1xbf16>
    %dw_b = "tosa.const"() <{values = dense<9.375000e-02> : tensor<48xbf16>}> : () -> tensor<48xbf16>
    %zp = arith.constant dense<0.000000e+00> : tensor<1xbf16>
    %conv = tosa.conv2d %arg0, %conv_w, %conv_b, %zp, %zp {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 0, 1, 0, 1>, stride = array<i64: 2, 2>} : (tensor<1x12x1280x24xbf16>, tensor<48x3x3x24xbf16>, tensor<48xbf16>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x6x640x48xbf16>
    %dw = tosa.depthwise_conv2d %conv, %dw_w, %dw_b, %zp, %zp {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x6x640x48xbf16>, tensor<3x3x48x1xbf16>, tensor<48xbf16>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x6x640x48xbf16>
    return %dw : tensor<1x6x640x48xbf16>
  }
}
