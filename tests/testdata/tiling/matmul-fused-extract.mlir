// Even an unused extract in a fused producer must trigger the conservative
// tile-shape guard. Memory-fit lowering can remove it, but the heuristic sees
// the original producer body before that cleanup.
func.func @main(%a: tensor<800x800xbf16>, %w: tensor<800x800xi8>, %hidden: tensor<1xbf16>) -> tensor<800x800xbf16> {
  %c0 = arith.constant 0 : index
  %zero = arith.constant 0.0 : bf16
  %we = tensor.empty() : tensor<800x800xbf16>
  %dq = linalg.generic {indexing_maps = [affine_map<(k,n)->(k,n)>, affine_map<(k,n)->(k,n)>], iterator_types = ["parallel", "parallel"]} ins(%w : tensor<800x800xi8>) outs(%we : tensor<800x800xbf16>) attrs = {"torq-fuse-group" = [2], "torq-fuse-group-id" = 0 : i64} {
  ^bb0(%v: i8, %o: bf16):
    %unused = tensor.extract %hidden[%c0] : tensor<1xbf16>
    %f = arith.sitofp %v : i8 to bf16
    linalg.yield %f : bf16
  } -> tensor<800x800xbf16>
  %e = tensor.empty() : tensor<800x800xbf16>
  %z = linalg.fill {"torq-fuse-group" = [2], "torq-fuse-group-id" = 1 : i64} ins(%zero : bf16) outs(%e : tensor<800x800xbf16>) -> tensor<800x800xbf16>
  %r = linalg.matmul {"torq-fuse-group" = [2], "torq-fuse-group-id" = 2 : i64} ins(%a, %dq : tensor<800x800xbf16>, tensor<800x800xbf16>) outs(%z : tensor<800x800xbf16>) -> tensor<800x800xbf16>
  return %r : tensor<800x800xbf16>
}
