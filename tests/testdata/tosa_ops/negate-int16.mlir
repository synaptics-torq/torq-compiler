module {
  func.func @main(%arg0: tensor<8x13x9xi16>) -> tensor<8x13x9xi16> {
    %cst = arith.constant dense<0> : tensor<1xi16>
    %cst_0 = arith.constant dense<0> : tensor<1xi16>
    %0 = tosa.negate %arg0, %cst, %cst_0 : (tensor<8x13x9xi16>, tensor<1xi16>, tensor<1xi16>) -> tensor<8x13x9xi16>
    return %0 : tensor<8x13x9xi16>
  }
}

