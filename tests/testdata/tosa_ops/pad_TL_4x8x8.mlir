module {
  func.func @main(%input: tensor<1x8x8x4xi8>) -> (tensor<1x9x9x4xi8>) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    // Allocate output for first transpose: NHWC -> NCHW
    %empty_nchw = tensor.empty() : tensor<1x4x8x8xi8>
    %nchw = linalg.transpose
      ins(%input : tensor<1x8x8x4xi8>)
      outs(%empty_nchw : tensor<1x4x8x8xi8>)
      permutation = [0, 3, 1, 2]

    // Pad in NCHW: pad H and W by [1,0] respectively to get 9x9
    %pad_val = arith.constant 19 : i8
    %padded = tensor.pad %nchw low[0, 0, 1, 1] high[0, 0, 0, 0] {
      ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):
        tensor.yield %pad_val : i8
    } : tensor<1x4x8x8xi8> to tensor<1x4x9x9xi8>

    // Allocate output for second transpose: NCHW -> NHWC
    %empty_nhwc = tensor.empty() : tensor<1x9x9x4xi8>
    %nhwc_out = linalg.transpose
      ins(%padded : tensor<1x4x9x9xi8>)
      outs(%empty_nhwc : tensor<1x9x9x4xi8>)
      permutation = [0, 2, 3, 1]

    return %nhwc_out : tensor<1x9x9x4xi8>
  }
}

