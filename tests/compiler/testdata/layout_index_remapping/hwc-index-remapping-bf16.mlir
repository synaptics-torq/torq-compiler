// Pass-only fixture for test_layout_index_remapping.py, not a simulator test.
// A constant HWC bias generic enters the depthwise-convolution epilogue cluster.
// The test permutes its loop axes to cover every three-loop ordering.
func.func @layout_indices(
    %input: tensor<1x5x5x5xbf16>, %filter: tensor<1x1x5xbf16>
) -> tensor<1x5x5x5xbf16> {
  %zero = arith.constant 0.0 : bf16
  %empty = tensor.empty() : tensor<1x5x5x5xbf16>
  %init = linalg.fill ins(%zero : bf16) outs(%empty : tensor<1x5x5x5xbf16>)
    -> tensor<1x5x5x5xbf16>
  %conv = linalg.depthwise_conv_2d_nhwc_hwc {
    strides = dense<1> : vector<2xi64>, dilations = dense<1> : vector<2xi64>
  } ins(%input, %filter : tensor<1x5x5x5xbf16>, tensor<1x1x5xbf16>)
     outs(%init : tensor<1x5x5x5xbf16>) -> tensor<1x5x5x5xbf16>
  %bias_empty = tensor.empty() : tensor<5x5x5xbf16>
  %bias_init = linalg.fill ins(%zero : bf16) outs(%bias_empty : tensor<5x5x5xbf16>)
    -> tensor<5x5x5xbf16>
  %bias = linalg.generic {
    indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>,
                     affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
    iterator_types = ["parallel", "parallel", "parallel"]
  } ins(%bias_init : tensor<5x5x5xbf16>)
    outs(%bias_empty : tensor<5x5x5xbf16>) {
  ^bb0(%value: bf16, %unused: bf16):
    %i0 = linalg.index 0 : index
    %i1 = linalg.index 1 : index
    %i2 = linalg.index 2 : index
    %v0 = arith.index_cast %i0 : index to i32
    %v1 = arith.index_cast %i1 : index to i32
    %v2 = arith.index_cast %i2 : index to i32
    %c100 = arith.constant 100 : i32
    %c10 = arith.constant 10 : i32
    %weighted0 = arith.muli %v0, %c100 : i32
    %weighted1 = arith.muli %v1, %c10 : i32
    %sum01 = arith.addi %weighted0, %weighted1 : i32
    %sum = arith.addi %sum01, %v2 : i32
    %coordinate = arith.sitofp %sum : i32 to f32
    %value_f32 = arith.extf %value : bf16 to f32
    %combined = arith.addf %value_f32, %coordinate : f32
    %result = arith.truncf %combined : f32 to bf16
    linalg.yield %result : bf16
  } -> tensor<5x5x5xbf16>
  %out = tensor.empty() : tensor<1x5x5x5xbf16>
  %result = linalg.generic {
    indexing_maps = [affine_map<(n, h, w, c) -> (n, h, w, c)>,
                     affine_map<(n, h, w, c) -> (h, w, c)>,
                     affine_map<(n, h, w, c) -> (n, h, w, c)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv, %bias : tensor<1x5x5x5xbf16>, tensor<5x5x5xbf16>)
    outs(%out : tensor<1x5x5x5xbf16>) {
  ^bb0(%value: bf16, %bias_value: bf16, %unused: bf16):
    %value_f32 = arith.extf %value : bf16 to f32
    %bias_f32 = arith.extf %bias_value : bf16 to f32
    %sum = arith.addf %value_f32, %bias_f32 : f32
    %result = arith.truncf %sum : f32 to bf16
    linalg.yield %result : bf16
  } -> tensor<1x5x5x5xbf16>
  return %result : tensor<1x5x5x5xbf16>
}
