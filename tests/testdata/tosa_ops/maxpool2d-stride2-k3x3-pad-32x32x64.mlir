module {
  func.func @main(%arg0: tensor<1x32x32x64xi8>) -> (tensor<1x16x16x64xi8>) {
    %32 = "tosa.const"() <{value = dense<[[0, 0], [0, 1], [0, 1], [0, 0]]> : tensor<4x2xi32>}> : () -> tensor<4x2xi32>
    %114 = "tosa.const"() <{value = dense<-100> : tensor<i8>}> : () -> tensor<i8>
    %119 = tosa.pad %arg0, %32, %114 {quantization_info = #tosa.pad_quant<input_zp = -100>} : (tensor<1x32x32x64xi8>, tensor<4x2xi32>, tensor<i8>) -> tensor<1x33x33x64xi8>
    %120 = tosa.max_pool2d %119 {kernel = array<i64: 3, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x33x33x64xi8>) -> tensor<1x16x16x64xi8>
    return %120 : tensor<1x16x16x64xi8>
  }
}

