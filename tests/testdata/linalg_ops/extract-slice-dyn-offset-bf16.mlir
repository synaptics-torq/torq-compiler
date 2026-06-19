module {
  func.func @main(%input: tensor<1024xbf16>, %off_tensor: tensor<1xi32>) -> tensor<512xbf16> {
    %c0 = arith.constant 0 : index
    %off_i32 = tensor.extract %off_tensor[%c0] : tensor<1xi32>
    %off = arith.index_cast %off_i32 : i32 to index
    %s = tensor.extract_slice %input[%off] [512] [1] : tensor<1024xbf16> to tensor<512xbf16>
    %cst = arith.constant 0.0 : bf16
    %e = tensor.empty() : tensor<512xbf16>
    %f = linalg.fill ins(%cst : bf16) outs(%e : tensor<512xbf16>) -> tensor<512xbf16>
    %r = linalg.generic {
        indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
        iterator_types = ["parallel"]
    } ins(%s : tensor<512xbf16>) outs(%f : tensor<512xbf16>) {
    ^bb0(%in: bf16, %out: bf16):
      %m = arith.mulf %in, %in : bf16
      linalg.yield %m : bf16
    } -> tensor<512xbf16>
    return %r : tensor<512xbf16>
  }
}
