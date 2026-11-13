#identity = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#keepdims = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
#broadcast = affine_map<(d0, d1, d2, d3) -> (d0, 0, d2, d3)>
module {
  func.func @main(%input: tensor<1x1024x32x32xi8>) -> (tensor<1x1x32x32xi8>, tensor<1x1024x32x32xi8>) {
    %zero = arith.constant 0 : i8
    %sum_empty = tensor.empty() : tensor<1x1x32x32xi8>
    %sum_init = linalg.fill ins(%zero : i8) outs(%sum_empty : tensor<1x1x32x32xi8>) -> tensor<1x1x32x32xi8>
    %sum = linalg.generic {indexing_maps = [#identity, #keepdims], iterator_types = ["parallel", "reduction", "parallel", "parallel"], "torq-fuse-group" = [2 : i64], "torq-fuse-group-id" = 2 : i64} ins(%input : tensor<1x1024x32x32xi8>) outs(%sum_init : tensor<1x1x32x32xi8>) {
    ^bb0(%in: i8, %acc: i8):
      %next = arith.addi %in, %acc : i8
      linalg.yield %next : i8
    } -> tensor<1x1x32x32xi8>
    %result_empty = tensor.empty() : tensor<1x1024x32x32xi8>
    %result = linalg.generic {indexing_maps = [#identity, #broadcast, #identity], iterator_types = ["parallel", "parallel", "parallel", "parallel"], "torq-fuse-group" = [2 : i64], "torq-fuse-group-id" = 5 : i64} ins(%input, %sum : tensor<1x1024x32x32xi8>, tensor<1x1x32x32xi8>) outs(%result_empty : tensor<1x1024x32x32xi8>) {
    ^bb0(%in: i8, %shared: i8, %out: i8):
      %next = arith.addi %in, %shared : i8
      linalg.yield %next : i8
    } -> tensor<1x1024x32x32xi8>
    return %sum, %result : tensor<1x1x32x32xi8>, tensor<1x1024x32x32xi8>
  }
}
