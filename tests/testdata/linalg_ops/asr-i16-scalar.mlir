module  {
  func.func @main(%2: tensor<1x1024x64xi16>) -> tensor<1x1024x64xi16> attributes {} {
    %c1_i16 = arith.constant 1 : i16
    %5 = tensor.empty() : tensor<1x1024x64xi16>
    %6 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>], iterator_types = ["parallel", "parallel", "parallel"]} ins(%2 : tensor<1x1024x64xi16>) outs(%5 : tensor<1x1024x64xi16>) {
    ^bb0(%in: i16, %out: i16):
      %18 = arith.shrsi %in, %c1_i16 : i16
      linalg.yield %18 : i16
    } -> tensor<1x1024x64xi16>
    return %6 : tensor<1x1024x64xi16>
  }
}