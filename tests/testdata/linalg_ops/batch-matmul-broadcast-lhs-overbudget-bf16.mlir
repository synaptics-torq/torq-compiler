// A batch_matmul whose LHS is a batch-invariant [M,K] broadcast (the collapse rewrite
// needs a batch-invariant RHS), so it stays a broadcast generic. Its [4,64,1024] bf16
// output (512 KiB) alone exceeds the LRAM budget: reduction tiling can never fit it and
// must skip (a hard failure would also poison tile-and-fuse's fits-in-memory probe).
module {
  func.func @main(%arg0: tensor<64x64xbf16>, %arg1: tensor<4x64x1024xbf16>) -> (tensor<4x64x1024xbf16>) {
    %cst = arith.constant 0.000000e+00 : bf16
    %0 = tensor.empty() : tensor<4x64x1024xbf16>
    %init = linalg.fill ins(%cst : bf16) outs(%0 : tensor<4x64x1024xbf16>) -> tensor<4x64x1024xbf16>
    // Broadcast the LHS from [64,64] to [4,64,64].
    %bcast_init = tensor.empty() : tensor<4x64x64xbf16>
    %bcast = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d1, d2)>,
                       affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%arg0 : tensor<64x64xbf16>) outs(%bcast_init : tensor<4x64x64xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      linalg.yield %in : bf16
    } -> tensor<4x64x64xbf16>
    %result = linalg.batch_matmul ins(%bcast, %arg1 : tensor<4x64x64xbf16>, tensor<4x64x1024xbf16>) outs(%init : tensor<4x64x1024xbf16>) -> tensor<4x64x1024xbf16>
    return %result : tensor<4x64x1024xbf16>
  }
}
