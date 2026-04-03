module {
  func.func @main(%arg0: tensor<1x112x112x128xi8>) -> tensor<1x56x56x128xi8> {
    %0 = tosa.const_shape  {values = dense<[0, 0, 0, 1, 0, 1, 0, 0]> : tensor<8xindex>} : () -> !tosa.shape<8>
    %cst = arith.constant dense<-100> : tensor<1xi8>
    %1 = tosa.pad %arg0, %0, %cst {quantization_info = #tosa.pad_quant<input_zp = -100>} : (tensor<1x112x112x128xi8>, !tosa.shape<8>, tensor<1xi8>) -> tensor<1x113x113x128xi8>
    %2 = tosa.max_pool2d %1 {kernel = array<i64: 3, 3>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x113x113x128xi8>) -> tensor<1x56x56x128xi8>
    return %2 : tensor<1x56x56x128xi8>
  }
}

