// An elementwise LHS producer feeding a [4096,512]x[512,64] matmul. The [4096,512] LHS does
// not fit LRAM but the [512,64] weight does, so the matmul is tiled along M only. The LHS
// slice follows the M tile, so fusing the producer recomputes nothing: tile-and-fuse must
// keep it fused in the matmul's loop.
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @main(%x: tensor<4096x512xbf16>, %w: tensor<512x64xbf16>) -> tensor<4096x64xbf16> {
  %e = tensor.empty() : tensor<4096x512xbf16>
  %neg = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<4096x512xbf16>) outs(%e : tensor<4096x512xbf16>) {
  ^bb0(%in: bf16, %o: bf16):
    %n = arith.negf %in : bf16
    linalg.yield %n : bf16
  } -> tensor<4096x512xbf16>
  %zero = arith.constant 0.0 : bf16
  %e2 = tensor.empty() : tensor<4096x64xbf16>
  %f = linalg.fill ins(%zero : bf16) outs(%e2 : tensor<4096x64xbf16>) -> tensor<4096x64xbf16>
  %r = linalg.matmul ins(%neg, %w : tensor<4096x512xbf16>, tensor<512x64xbf16>) outs(%f : tensor<4096x64xbf16>) -> tensor<4096x64xbf16>
  return %r : tensor<4096x64xbf16>
}
