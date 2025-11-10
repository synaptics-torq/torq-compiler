module {
  func.func @main(%arg0: tensor<1x7x7x128xi8>) -> (tensor<1x1x1x128xi8>) {
    %261 = tosa.avg_pool2d %arg0 {acc_type = i32, kernel = array<i64: 7, 7>, pad = array<i64: 0, 0, 0, 0>, quantization_info = #tosa.unary_quant<input_zp = 4, output_zp = 0>, stride = array<i64: 1, 1>} : (tensor<1x7x7x128xi8>) -> tensor<1x1x1x128xi8>
    return %261 : tensor<1x1x1x128xi8>
  }
}

