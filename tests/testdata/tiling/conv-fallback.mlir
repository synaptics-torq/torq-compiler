// F=16 cannot fit even with H=W=1. The fallback reduces F below its slice
// floor; legacy grow-back must leave the first pass's H=W=1 untouched.
func.func @main(%a: tensor<1x32768x2x2xbf16>, %w: tensor<16x32768x1x1xbf16>) -> tensor<1x16x2x2xbf16> {
  %zero = arith.constant 0.0 : bf16
  %e = tensor.empty() : tensor<1x16x2x2xbf16>
  %z = linalg.fill ins(%zero : bf16) outs(%e : tensor<1x16x2x2xbf16>) -> tensor<1x16x2x2xbf16>
  %r = linalg.conv_2d_nchw_fchw {dilations = dense<1> : tensor<2xi64>, strides = dense<1> : tensor<2xi64>} ins(%a, %w : tensor<1x32768x2x2xbf16>, tensor<16x32768x1x1xbf16>) outs(%z : tensor<1x16x2x2xbf16>) -> tensor<1x16x2x2xbf16>
  return %r : tensor<1x16x2x2xbf16>
}
