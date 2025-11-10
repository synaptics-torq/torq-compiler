module {
  func.func @main(%arg0: tensor<1x32x32x4xi8>) -> (tensor<1x32x32x4xi8>) {
    %0 = tosa.max_pool2d %arg0 {quantization_info = #tosa.conv_quant<input_zp = -128, weight_zp = 0>, kernel = array<i64: 3, 3>, pad = array<i64: 1, 1, 1, 1>, stride = array<i64: 1, 1>} : (tensor<1x32x32x4xi8>) -> tensor<1x32x32x4xi8>
    return %0 : tensor<1x32x32x4xi8>
  }
}

