// Pass-only regression: preserve all N/H/W/C queries when NHWC becomes NCHW.
func.func @preserve_nhwc_indices(
    %input: tensor<1x3x16x7xbf16>, %filter: tensor<1x1x7xbf16>
) -> tensor<1x3x16x7xbf16> {
  %zero = arith.constant 0.0 : bf16
  %empty = tensor.empty() : tensor<1x3x16x7xbf16>
  %init = linalg.fill ins(%zero : bf16)
    outs(%empty : tensor<1x3x16x7xbf16>) -> tensor<1x3x16x7xbf16>
  %conv = linalg.depthwise_conv_2d_nhwc_hwc {
    strides = dense<1> : vector<2xi64>, dilations = dense<1> : vector<2xi64>
  } ins(%input, %filter : tensor<1x3x16x7xbf16>, tensor<1x1x7xbf16>)
    outs(%init : tensor<1x3x16x7xbf16>) -> tensor<1x3x16x7xbf16>
  %out = tensor.empty() : tensor<1x3x16x7xbf16>
  %result = linalg.generic {
    indexing_maps = [affine_map<(n, h, w, c) -> (n, h, w, c)>,
                     affine_map<(n, h, w, c) -> (n, h, w, c)>],
    iterator_types = ["parallel", "parallel", "parallel", "parallel"]
  } ins(%conv : tensor<1x3x16x7xbf16>)
    outs(%out : tensor<1x3x16x7xbf16>) {
  ^bb0(%value: bf16, %unused: bf16):
    %n = linalg.index 0 : index
    %h = linalg.index 1 : index
    %w = linalg.index 2 : index
    %c = linalg.index 3 : index
    %ni = arith.index_cast %n : index to i32
    %hi = arith.index_cast %h : index to i32
    %wi = arith.index_cast %w : index to i32
    %ci = arith.index_cast %c : index to i32
    %c1000 = arith.constant 1000 : i32
    %c100 = arith.constant 100 : i32
    %c10 = arith.constant 10 : i32
    %nw = arith.muli %ni, %c1000 : i32
    %hw = arith.muli %hi, %c100 : i32
    %ww = arith.muli %wi, %c10 : i32
    %nh = arith.addi %nw, %hw : i32
    %nhw = arith.addi %nh, %ww : i32
    %nhwc = arith.addi %nhw, %ci : i32
    %coordinate = arith.sitofp %nhwc : i32 to f32
    %wide = arith.extf %value : bf16 to f32
    %sum = arith.addf %wide, %coordinate : f32
    %narrow = arith.truncf %sum : f32 to bf16
    linalg.yield %narrow : bf16
  } -> tensor<1x3x16x7xbf16>
  return %result : tensor<1x3x16x7xbf16>
}
