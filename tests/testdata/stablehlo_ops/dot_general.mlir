module @jit___call  {
  func.func @main(%4374: tensor<1x256x4x256xbf16>, %4375: tensor<1x256x256xbf16>) -> tensor<1x256x4x256xbf16> attributes {} {
    %4376 = stablehlo.dot_general %4374, %4375, batching_dims = [0] x [0], contracting_dims = [3] x [1], precision = [DEFAULT, DEFAULT] : (tensor<1x256x4x256xbf16>, tensor<1x256x256xbf16>) -> tensor<1x256x4x256xbf16>
    return %4376 : tensor<1x256x4x256xbf16>
  }
}
