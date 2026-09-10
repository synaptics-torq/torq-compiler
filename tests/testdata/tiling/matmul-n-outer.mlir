// The [300,800] LHS fits LRAM whole while the [800,3200] weight does not, and
// M still needs tiling: n-outer streams the weight once instead of once per M
// tile, so the loop-order model must reverse to for(n)for(m).
func.func @main(%a: tensor<300x800xbf16>, %w: tensor<800x3200xbf16>) -> tensor<300x3200xbf16> {
  %zero = arith.constant 0.0 : bf16
  %e = tensor.empty() : tensor<300x3200xbf16>
  %z = linalg.fill {"torq-fuse-group" = [1], "torq-fuse-group-id" = 0 : i64} ins(%zero : bf16) outs(%e : tensor<300x3200xbf16>) -> tensor<300x3200xbf16>
  %r = linalg.matmul {"torq-fuse-group" = [1], "torq-fuse-group-id" = 1 : i64} ins(%a, %w : tensor<300x800xbf16>, tensor<800x3200xbf16>) outs(%z : tensor<300x3200xbf16>) -> tensor<300x3200xbf16>
  return %r : tensor<300x3200xbf16>
}
