module {
  func.func @main(%arg0: tensor<1x1024x64xi8>) -> tensor<1x1024x64xbf16> {
    %init = tensor.empty() : tensor<1x1024x64xbf16>
    %0 = linalg.generic {
      indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1, d2)>],
      iterator_types = ["parallel", "parallel", "parallel"]
    } ins(%arg0 : tensor<1x1024x64xi8>) outs(%init : tensor<1x1024x64xbf16>) {
    ^bb0(%in: i8, %out: bf16):
      %c = arith.uitofp %in : i8 to bf16
      linalg.yield %c : bf16
    } -> tensor<1x1024x64xbf16>
    return %0 : tensor<1x1024x64xbf16>
  }
}
