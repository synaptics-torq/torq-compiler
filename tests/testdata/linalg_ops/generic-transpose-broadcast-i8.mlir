// [#1760] A copy-only linalg.generic whose input map both permutes the kept
// axes (d1,d2 swapped) and drops one (broadcast over d3). It is neither a pure
// broadcast (isaBroadcastOpInterface needs monotonically-increasing dims) nor a
// linalg.transpose, and must lower to torq_hl.transpose + torq_hl.broadcast.
module {
  func.func @main(%arg0: tensor<1x4x3xi8>) -> (tensor<1x3x4x5xi8>) {
    %init = tensor.empty() : tensor<1x3x4x5xi8>
    %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d2, d1)>,
                       affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>],
      iterator_types = ["parallel", "parallel", "parallel", "parallel"]
    } ins(%arg0 : tensor<1x4x3xi8>) outs(%init : tensor<1x3x4x5xi8>) {
      ^bb0(%a: i8, %b: i8):
        linalg.yield %a : i8
    } -> tensor<1x3x4x5xi8>
    return %0 : tensor<1x3x4x5xi8>
  }
}
