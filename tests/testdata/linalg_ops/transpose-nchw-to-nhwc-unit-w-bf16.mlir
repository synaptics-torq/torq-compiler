// Regression: an NCHW->NHWC transpose of [1,C,H,1] with C == H is a real 16x16
// transpose; the data-preserving transpose fold must not turn it into a reshape.
module {
  func.func @main(%arg0: tensor<1x16x16x1xbf16>) -> tensor<1x16x1x16xbf16> {
    %init = tensor.empty() : tensor<1x16x1x16xbf16>
    %0 = linalg.transpose ins(%arg0 : tensor<1x16x16x1xbf16>) outs(%init : tensor<1x16x1x16xbf16>) permutation = [0, 2, 3, 1]
    return %0 : tensor<1x16x1x16xbf16>
  }
}
