// An elementwise LHS producer feeding a [256,512]x[512,8192] matmul. The [512,8192] weight
// does not fit LRAM, so the matmul is tiled along N. The LHS slice does not index N, so
// fusing the producer would recompute it for every N tile: tile-and-fuse must leave it in
// its own loop, computed once.
#map = affine_map<(d0, d1) -> (d0, d1)>
func.func @main(%x: tensor<256x512xbf16>, %w: tensor<512x8192xbf16>) -> tensor<256x8192xbf16> {
  %e = tensor.empty() : tensor<256x512xbf16>
  %neg = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%x : tensor<256x512xbf16>) outs(%e : tensor<256x512xbf16>) {
  ^bb0(%in: bf16, %o: bf16):
    %n = arith.negf %in : bf16
    linalg.yield %n : bf16
  } -> tensor<256x512xbf16>
  %zero = arith.constant 0.0 : bf16
  %e2 = tensor.empty() : tensor<256x8192xbf16>
  %f = linalg.fill ins(%zero : bf16) outs(%e2 : tensor<256x8192xbf16>) -> tensor<256x8192xbf16>
  %r = linalg.matmul ins(%neg, %w : tensor<256x512xbf16>, tensor<512x8192xbf16>) outs(%f : tensor<256x8192xbf16>) -> tensor<256x8192xbf16>
  return %r : tensor<256x8192xbf16>
}
