module {
  func.func @main(%arg0: tensor<4x1x256xbf16>, %arg1: tensor<1x256x256xbf16>) -> (tensor<4x1x256xbf16>) {
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<4x1x256xbf16>
    %init = linalg.fill ins(%cst : bf16) outs(%0 : tensor<4x1x256xbf16>) -> tensor<4x1x256xbf16>
    // Broadcast B from [1,256,256] to [4,256,256]
    // The compiler should eliminate the explicit broadcast and use stride=0
    %bcast_init = tensor.empty() : tensor<4x256x256xbf16>
    %bcast = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (0, d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%arg1 : tensor<1x256x256xbf16>) outs(%bcast_init : tensor<4x256x256xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4x256x256xbf16>
    %result = linalg.batch_matmul ins(%arg0, %bcast : tensor<4x1x256xbf16>, tensor<4x256x256xbf16>) outs(%init : tensor<4x1x256xbf16>) -> tensor<4x1x256xbf16>
    return %result : tensor<4x1x256xbf16>
  }
}
