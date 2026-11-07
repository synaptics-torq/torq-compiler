module {
  func.func @main(%arg0: tensor<1x192x10x10xbf16>) -> tensor<1x192x10x10xbf16> {
    // NCHW -> NHWC
    %nhwc = tosa.transpose %arg0 perms([0, 2, 3, 1]) 
      : (tensor<1x192x10x10xbf16>) -> tensor<1x10x10x192xbf16>

    // pad = [top, bottom, left, right]
    %pooled = tosa.max_pool2d %nhwc kernel([5, 5]) stride([1, 1]) pad([2, 2, 2, 2]) {acc_type = f32} : (tensor<1x10x10x192xbf16>) -> tensor<1x10x10x192xbf16>

    // NHWC -> NCHW
    %out = tosa.transpose %pooled perms([0, 3, 1, 2]) 
      : (tensor<1x10x10x192xbf16>) -> tensor<1x192x10x10xbf16>

    return %out : tensor<1x192x10x10xbf16>
  }
}