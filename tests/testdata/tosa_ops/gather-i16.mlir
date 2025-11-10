module {
  func.func @main(%arg0: tensor<1x256x32xi16>) -> (tensor<1x16x32xi16>) {
    %0 = "tosa.const"() <{
      value = dense<[[1, 2, 3, 4, 5, 6, 7, 78, 90, 10, 22, 55, 15, 128, 222, 202]]> :
        tensor<1x16xi32>
    }> : () -> tensor<1x16xi32>

    %1 = tosa.gather %arg0, %0
         : (tensor<1x256x32xi16>, tensor<1x16xi32>) -> tensor<1x16x32xi16>
    return %1 : tensor<1x16x32xi16>
  }
}
