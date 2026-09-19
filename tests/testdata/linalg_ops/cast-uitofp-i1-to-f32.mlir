module {
  func.func @main(%arg0: tensor<1x1024x64xi1>) -> tensor<1x1024x64xf32> {
    %init = tensor.empty() : tensor<1x1024x64xf32>
    %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%arg0 : tensor<1x1024x64xi1>) outs(%init : tensor<1x1024x64xf32>) {
    ^bb0(%in: i1, %out: f32):
      %c = arith.uitofp %in : i1 to f32
      linalg.yield %c : f32
    } -> tensor<1x1024x64xf32>
    return %0 : tensor<1x1024x64xf32>
  }
}
