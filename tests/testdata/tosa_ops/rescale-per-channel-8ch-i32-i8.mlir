module {
  func.func @main(%arg0: tensor<1x2x4x8xi32>) -> tensor<1x2x4x8xi8> attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %mult = arith.constant dense<[536870912, 805306368, 1073741824, 1342177280, 1610612736, 1879048192, 2147483647, 671088640]> : tensor<8xi32>
    %shift = arith.constant dense<30> : tensor<8xi8>
    %in_zp = arith.constant dense<0> : tensor<1xi32>
    %out_zp = arith.constant dense<3> : tensor<1xi8>
    %0 = tosa.rescale %arg0, %mult, %shift, %in_zp, %out_zp {input_unsigned = false, output_unsigned = false, per_channel = true, rounding_mode = DOUBLE_ROUND, scale32 = true} : (tensor<1x2x4x8xi32>, tensor<8xi32>, tensor<8xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x2x4x8xi8>
    return %0 : tensor<1x2x4x8xi8>
  }
}
