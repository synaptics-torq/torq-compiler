module  {
  func.func @main(%2: tensor<1x1024x64xi8>) -> tensor<1x1024x64xi8> attributes {} {
    %c1_i8 = arith.constant 3 : i8
    %5 = tensor.empty() : tensor<1x1024x64xi8>
    %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x1024x64xi8>) outs(%5 : tensor<1x1024x64xi8>) {
    ^bb0(%in: i8, %out: i8):
      %18 = arith.shrsi %in, %c1_i8 : i8
      linalg.yield %18 : i8
    } -> tensor<1x1024x64xi8>
    return %6 : tensor<1x1024x64xi8>
  }
}