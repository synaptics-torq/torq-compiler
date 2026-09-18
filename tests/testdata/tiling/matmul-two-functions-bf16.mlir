// Regression input for synaptics-torq/torq-compiler-dev#2351 (probe diagnostic
// handler thread race). Two exported functions with independent tosa.matmul
// ops mimic a prefill/decode pair: @main's weight (1024x1024x2 B = 2 MB) is
// larger than SL2610's LRAM (523,008 B), so the tile-fit memory probe must
// reject it while tile-and-fuse runs both functions' probes concurrently.
module {
  func.func @main(%arg0: tensor<1x128x1024xbf16>, %arg1: tensor<1x1024x1024xbf16>) -> tensor<1x128x1024xf32> attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %cst = arith.constant dense<0.000000e+00> : tensor<1xbf16>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1xbf16>
    %0 = tosa.matmul %arg0, %arg1, %cst, %cst_0 : (tensor<1x128x1024xbf16>, tensor<1x1024x1024xbf16>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x128x1024xf32>
    return %0 : tensor<1x128x1024xf32>
  }
  func.func @main2(%arg0: tensor<1x16x1024xbf16>, %arg1: tensor<1x1024x1024xbf16>) -> tensor<1x16x1024xf32> attributes {tf_saved_model.exported_names = ["serving_default2"]} {
    %cst = arith.constant dense<0.000000e+00> : tensor<1xbf16>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1xbf16>
    %0 = tosa.matmul %arg0, %arg1, %cst, %cst_0 : (tensor<1x16x1024xbf16>, tensor<1x1024x1024xbf16>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x16x1024xf32>
    return %0 : tensor<1x16x1024xf32>
  }
}
