module attributes {tf_saved_model.semantics} {
  func.func @main(%1470: tensor<1024xi1>) -> tensor<1024xi8> {
  %cst = arith.constant 5 : i8
  %cst_0 = arith.constant 3 : i8
  %c0 = arith.constant 0 : index
  %4 = tensor.empty() : tensor<1024xi8>
  %5 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%1470 : tensor<1024xi1>) outs(%4 : tensor<1024xi8>) {
    ^bb0(%in: i1, %out: i8):
      %6 = arith.select %in, %cst, %cst_0 : i8
      linalg.yield %6 : i8
    } -> tensor<1024xi8>
  return %5 : tensor<1024xi8>
  }
}
