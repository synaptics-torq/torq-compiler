// i1 elementwise And with a non-constant broadcast operand — the decoder
// causal-mask shape ([1,1,8,8] against a [1,1,1,8] padding mask).
module {
  func.func @main(%arg0: tensor<1x1x8x8xi1>, %arg1: tensor<1x1x1x8xi1>) -> tensor<1x1x8x8xi1> {
    %0 = tensor.empty() : tensor<1x1x8x8xi1>
    %1 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, 0, d3)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%arg0, %arg1 : tensor<1x1x8x8xi1>, tensor<1x1x1x8xi1>)
      outs(%0 : tensor<1x1x8x8xi1>) {
      ^bb0(%a: i1, %b: i1, %o: i1):
        %r = arith.andi %a, %b : i1
        linalg.yield %r : i1
    } -> tensor<1x1x8x8xi1>
    return %1 : tensor<1x1x8x8xi1>
  }
}
