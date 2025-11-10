module {
  func.func @main(%arg0: tensor<1x128x64xi8>, %arg1: tensor<1x64x1xi8>) -> (tensor<1x128x1xi32>) {
    %0 = tosa.matmul %arg0, %arg1 {quantization_info = #tosa.matmul_quant<a_zp = 0, b_zp = 0>} : (tensor<1x128x64xi8>, tensor<1x64x1xi8>) -> tensor<1x128x1xi32>
    return %0 : tensor<1x128x1xi32>
  }
}

