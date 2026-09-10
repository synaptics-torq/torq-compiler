// Explicit pattern group keeps the live int8 cast in the matmul tile.
// With 100x192 tiles, 8-bit B selects M outermost; 16-bit B would reverse it.
func.func @main(%a: tensor<800x800xbf16>, %w: tensor<800x800xi8>) -> tensor<800x800xbf16> {
  %zero = arith.constant 0.0 : bf16
  %we = tensor.empty() : tensor<800x800xbf16>
  %dq = linalg.generic {indexing_maps = [affine_map<(k,n)->(k,n)>, affine_map<(k,n)->(k,n)>], iterator_types = ["parallel", "parallel"]} ins(%w : tensor<800x800xi8>) outs(%we : tensor<800x800xbf16>) attrs = {"torq-fuse-group" = [2], "torq-fuse-group-id" = 0 : i64} {
  ^bb0(%v: i8, %o: bf16):
    %f = arith.sitofp %v : i8 to bf16
    linalg.yield %f : bf16
  } -> tensor<800x800xbf16>
  %e = tensor.empty() : tensor<800x800xbf16>
  %z = linalg.fill {"torq-fuse-group" = [2], "torq-fuse-group-id" = 1 : i64} ins(%zero : bf16) outs(%e : tensor<800x800xbf16>) -> tensor<800x800xbf16>
  %r = linalg.matmul {"torq-fuse-group" = [2], "torq-fuse-group-id" = 2 : i64} ins(%a, %dq : tensor<800x800xbf16>, tensor<800x800xbf16>) outs(%z : tensor<800x800xbf16>) -> tensor<800x800xbf16>
  return %r : tensor<800x800xbf16>
}
