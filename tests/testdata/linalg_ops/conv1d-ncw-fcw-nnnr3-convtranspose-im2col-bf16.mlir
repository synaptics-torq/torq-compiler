// Regression: NNNR3 ConvTranspose conv1d under NSS-only compilation
// (--torq-disable-host --torq-disable-css by the linalg_ops harness).
//
// Shapes from NNNR3_0079_0.0960_torq_bf16 layer ConvTranspose_52: ConvTranspose
// (stride 2, Kw=3, 48->48) lowers to stride-1 conv1d on a padded [1, 48, 35] input
// with Ow=33. The conv1d promotes to conv2d (unit H extent, 3x1 kernel) and
// lowers through the 2D conv path.
//
// Conv output is f32 (as on the real layer before bias/trunc); epilogue is omitted
// here because FC+trunc fusion needs embedded constants. Full layer coverage:
// test_onnx_model.py -k ConvTranspose. Filter is a function argument.
module {
  func.func @main(
      %arg0: tensor<1x48x35xbf16>, %arg1: tensor<48x48x3xbf16>
  ) -> tensor<1x48x33xf32> {
    %cst = arith.constant dense<0.000000e+00> : tensor<1x48x33xf32>
    %0 = linalg.conv_1d_ncw_fcw {dilations = dense<1> : vector<1xi64>, strides = dense<1> : vector<1xi64>}
        ins(%arg0, %arg1 : tensor<1x48x35xbf16>, tensor<48x48x3xbf16>)
        outs(%cst : tensor<1x48x33xf32>) -> tensor<1x48x33xf32>
    return %0 : tensor<1x48x33xf32>
  }
}
