module attributes {tf_saved_model.semantics} {
  func.func @main(%cond: tensor<1x1x1x1024xi1>) -> tensor<1x1x1x1024xf32> {
    %init = tensor.empty() : tensor<1x1x1x1024xf32>
    %result = linalg.generic
      { indexing_maps = [
          affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>, // cond
          affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>  // out
        ],
        iterator_types = ["parallel","parallel","parallel","parallel"] }
      ins(%cond : tensor<1x1x1x1024xi1>) outs(%init : tensor<1x1x1x1024xf32>) {
      ^bb0(%c: i1, %o: f32):
        %true = arith.constant 5.0 : f32
        %false = arith.constant 3.0 : f32
        %sel = arith.select %c, %true, %false : i1, f32
        linalg.yield %sel : f32
      } -> tensor<1x1x1x1024xf32>
    return %result : tensor<1x1x1x1024xf32>
  }
}
