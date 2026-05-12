module @jit___call  {
  func.func @main(%arg107: tensor<2048x640xbf16>, %4415: tensor<1x256x2x2048xbf16>, %4417: tensor<1x256x2048xbf16>, %4422: tensor<1x256x2048xbf16>) -> tensor<1x256x640xf32> attributes {} {
    %cst_996 = stablehlo.constant dense<7.968750e-01> : tensor<bf16>
    %4423 = stablehlo.broadcast_in_dim %cst_996, dims = [] : (tensor<bf16>) -> tensor<1x256x2048xbf16>
    %4424 = stablehlo.multiply %4423, %4422 : tensor<1x256x2048xbf16>
    %4425 = stablehlo.tanh %4424 : tensor<1x256x2048xbf16>
    %cst_997 = stablehlo.constant dense<1.000000e+00> : tensor<bf16>
    %4426 = stablehlo.broadcast_in_dim %cst_997, dims = [] : (tensor<bf16>) -> tensor<1x256x2048xbf16>
    %4427 = stablehlo.add %4426, %4425 : tensor<1x256x2048xbf16>
    %cst_998 = stablehlo.constant dense<5.000000e-01> : tensor<bf16>
    %4428 = stablehlo.broadcast_in_dim %cst_998, dims = [] : (tensor<bf16>) -> tensor<1x256x2048xbf16>
    %4429 = stablehlo.multiply %4428, %4427 : tensor<1x256x2048xbf16>
    %4430 = stablehlo.multiply %4417, %4429 : tensor<1x256x2048xbf16>
    %4431 = stablehlo.slice %4415 [0:1, 0:256, 1:2, 0:2048] : (tensor<1x256x2x2048xbf16>) -> tensor<1x256x1x2048xbf16>
    %4432 = stablehlo.reshape %4431 : (tensor<1x256x1x2048xbf16>) -> tensor<1x256x2048xbf16>
    %4433 = stablehlo.multiply %4430, %4432 : tensor<1x256x2048xbf16>
    %4434 = stablehlo.dot_general %4433, %arg107, contracting_dims = [2] x [0], precision = [DEFAULT, DEFAULT] : (tensor<1x256x2048xbf16>, tensor<2048x640xbf16>) -> tensor<1x256x640xbf16>
    %4435 = chlo.square %4434 : tensor<1x256x640xbf16> -> tensor<1x256x640xbf16>
    %4436 = stablehlo.convert %4435 : (tensor<1x256x640xbf16>) -> tensor<1x256x640xf32>
    return %4436 : tensor<1x256x640xf32>
  }
}
