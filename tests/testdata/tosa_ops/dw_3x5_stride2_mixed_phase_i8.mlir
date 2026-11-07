module {
  func.func @main(%arg0: tensor<1x32x32x8xi8> {ml_program.identifier = "serving_default_keras_tensor:0", tf_saved_model.index_path = ["keras_tensor"]}) -> (tensor<1x16x15x8xi8> {ml_program.identifier = "PartitionedCall_1:0", tf_saved_model.index_path = ["output_0"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    // Non-square (3x5) stride-2 quantized depthwise conv with mixed H/W pad phase
    // (H centered, W off-center): the mixed-phase normalization must handle it on NSS.
    %bias = "tosa.const"() <{values = dense<0> : tensor<8xi32>}> : () -> tensor<8xi32>
    %weight = "tosa.const"() <{values = dense<[[[[-8], [-52], [11], [76], [-78], [-72], [47], [-66]], [[3], [59], [-76], [39], [-36], [-81], [-68], [21]], [[17], [-73], [-29], [-67], [51], [18], [-75], [54]], [[-59], [-33], [71], [70], [59], [-75], [57], [59]], [[11], [-78], [-34], [-79], [52], [-56], [-16], [17]]], [[[-54], [48], [-60], [56], [-12], [53], [84], [-44]], [[-64], [58], [56], [73], [-42], [5], [-66], [50]], [[-74], [54], [-75], [68], [-38], [37], [84], [46]], [[19], [-10], [29], [59], [26], [2], [-14], [-27]], [[-44], [88], [-28], [-70], [57], [-14], [44], [36]]], [[[-3], [24], [-17], [65], [-72], [-60], [41], [17]], [[-48], [-3], [-52], [35], [17], [-80], [81], [-71]], [[52], [56], [-10], [-3], [87], [-1], [62], [37]], [[58], [26], [-73], [-67], [-21], [31], [88], [80]], [[-74], [-75], [89], [-11], [75], [57], [84], [24]]]]> : tensor<3x5x8x1xi8>}> : () -> tensor<3x5x8x1xi8>
    %zp_in = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %zp_out = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
    %1 = tosa.depthwise_conv2d %arg0, %weight, %bias, %zp_in, %zp_out pad([1, 0, 0, 1]) stride([2, 2]) dilation([1, 1]) acc_type(i32) : (tensor<1x32x32x8xi8>, tensor<3x5x8x1xi8>, tensor<8xi32>, tensor<1xi8>, tensor<1xi8>) -> tensor<1x16x15x8xi32>
    %mult = "tosa.const"() <{values = dense<1073741824> : tensor<8xi32>}> : () -> tensor<8xi32>
    %shift = "tosa.const"() <{values = dense<0> : tensor<8xi8>}> : () -> tensor<8xi8>
    %zp_out32 = "tosa.const"() <{values = dense<0> : tensor<1xi32>}> : () -> tensor<1xi32>
    %2 = tosa.rescale %1, %mult, %shift, %zp_out32, %zp_out scale32(true) rounding_mode<DOUBLE_ROUND> per_channel(true) input_unsigned(false) output_unsigned(false) : (tensor<1x16x15x8xi32>, tensor<8xi32>, tensor<8xi8>, tensor<1xi32>, tensor<1xi8>) -> tensor<1x16x15x8xi8>
    return %2 : tensor<1x16x15x8xi8>
  }
}
