module {
  func.func @main(%40: tensor<64xbf16>) -> tensor<64xbf16> {
    %cst = arith.constant 2.000000e+00 : bf16
    %3 = tensor.empty() : tensor<64xbf16>
    %4 = linalg.generic {indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>], iterator_types = ["parallel"]} ins(%40 : tensor<64xbf16>) outs(%3 : tensor<64xbf16>) attrs =  {"torq-executor" = #torq_hl.executor<host>, "torq-fuse-group-id" = 0 : i64} {
    ^bb0(%in: bf16, %out: bf16):
        %5 = math.powf %in, %cst : bf16
        linalg.yield %5 : bf16
    } -> tensor<64xbf16>
    return %4 : tensor<64xbf16>
  }
}
