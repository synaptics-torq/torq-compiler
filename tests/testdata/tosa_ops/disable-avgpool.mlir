// disable this test because hw kernel not support the real avgpool2d op

module {
  func.func @main(%arg0: tensor<1x7x7x1280xi8>) -> (tensor<1x1280xi8>) {
    %249 = tosa.avg_pool2d %arg0 {acc_type = i32, kernel = array<i64: 7, 7>, pad = array<i64: 0, 0, 0, 0>, quantization_info = #tosa.unary_quant<input_zp = 0, output_zp = 0>, stride = array<i64: 1, 1>} : (tensor<1x7x7x1280xi8>) -> tensor<1x1x1x1280xi8>
    %250 = tosa.reshape %249 {new_shape = array<i64: 1, 1280>} : (tensor<1x1x1x1280xi8>) -> tensor<1x1280xi8>
    return %250 : tensor<1x1280xi8>
  }
}

