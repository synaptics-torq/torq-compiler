// Regression: same fused bf16 fully_connected, with IC=66 so the reduction length is
// not a multiple of wram.transposeWidth(). maxDivisor(66, 8) picks 6, so the chunk is
// smaller than the transpose width and still divides IC.
//
// The weight is an argument, not a constant, purely to keep this file small.
#map = affine_map<(d0) -> (d0)>
#map1 = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<16x66xbf16>, %arg1: tensor<66x64xbf16>) -> tensor<16x64xbf16> {
    %cst = arith.constant dense<"0x45BE2E3F4BBE93BD9BBE78BE103F09BE03BFC7BF93BE7CBF69BFCC3EB53E75BFDA3E6B3EC9BEBC3E50BF1ABF38BF0EBFE13E2BBF13BFF4BEC83D2EBEDD3EB6BE25BF81BE8F3CEFBE0C3E683F97BE953C2C3FA8BEDCBE0EBF453E803EA7BD5A3C213FDA3E36BF72BC2C3DB33EFCBDABBE77BE67BFD6BCA0BE603E07BF6EBE01BF"> : tensor<64xbf16>
    %0 = tensor.empty() : tensor<16x64xf32>
    %1 = tensor.empty() : tensor<64xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel"]} ins(%cst : tensor<64xbf16>) outs(%1 : tensor<64xf32>) {
    ^bb0(%in: bf16, %out: f32):
      %7 = arith.extf %in : bf16 to f32
      linalg.yield %7 : f32
    } -> tensor<64xf32>
    %broadcasted = linalg.broadcast ins(%2 : tensor<64xf32>) outs(%0 : tensor<16x64xf32>) dimensions = [0]
    %3 = linalg.matmul ins(%arg0, %arg1 : tensor<16x66xbf16>, tensor<66x64xbf16>) outs(%broadcasted : tensor<16x64xf32>) -> tensor<16x64xf32>
    %4 = tensor.empty() : tensor<16x64xbf16>
    %5 = linalg.generic {indexing_maps = [#map1, #map1], iterator_types = ["parallel", "parallel"]} ins(%3 : tensor<16x64xf32>) outs(%4 : tensor<16x64xbf16>) {
    ^bb0(%in: f32, %out: bf16):
      %7 = arith.truncf %in : f32 to bf16
      linalg.yield %7 : bf16
    } -> tensor<16x64xbf16>
    return %5 : tensor<16x64xbf16>
  }
}
