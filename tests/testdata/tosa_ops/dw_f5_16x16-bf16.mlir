module attributes {tf_saved_model.semantics} {
  func.func @main(%arg0: tensor<1x16x16x2xbf16> {ml_program.identifier = "serving_default_input_0:0", tf_saved_model.index_path = ["input_0"]}) -> (tensor<1x16x16x2xbf16> {ml_program.identifier = "StatefulPartitionedCall_1:0", tf_saved_model.index_path = ["output_0"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tosa.const"() <{values = dense<[1.100000e+01, 2.200000e+01]> : tensor<2xbf16>}> : () -> tensor<2xbf16>
    %1 = "tosa.const"() <{values = dense<[[[[1.500000e+01], [7.000000e+00]], [[1.800000e+01], [8.000000e+00]], [[1.200000e+01], [8.500000e+00]], [[1.500000e+01], [1.800000e+01]], [[1.700000e+01], [1.250000e+01]]], [[[1.400000e+01], [4.000000e+00]], [[9.000000e+00], [9.000000e+00]], [[1.000000e+01], [9.000000e+00]], [[2.000000e+00], [4.500000e+00]], [[8.000000e+00], [6.000000e+00]]], [[[1.600000e+01], [9.500000e+00]], [[1.100000e+01], [6.000000e+00]], [[1.925000e+01], [7.000000e+00]], [[1.600000e+01], [6.000000e+00]], [[1.000000e+00], [3.500000e+00]]], [[[1.500000e+01], [7.000000e+00]], [[1.800000e+01], [8.000000e+00]], [[1.200000e+01], [8.500000e+00]], [[5.000000e+00], [1.800000e+01]], [[1.700000e+01], [1.500000e+00]]], [[[1.500000e+01], [1.700000e+01]], [[1.800000e+01], [8.000000e+00]], [[1.200000e+01], [1.850000e+01]], [[1.000000e+01], [1.800000e+01]], [[1.700000e+01], [1.250000e+01]]]]> : tensor<5x5x2x1xbf16>}> : () -> tensor<5x5x2x1xbf16>
    %cst = arith.constant dense<0.000000e+00> : tensor<1xbf16>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<1xbf16>
    %2 = tosa.depthwise_conv2d %arg0, %1, %0, %cst, %cst_0 {acc_type = f32, dilation = array<i64: 1, 1>, pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 1, 1>} : (tensor<1x16x16x2xbf16>, tensor<5x5x2x1xbf16>, tensor<2xbf16>, tensor<1xbf16>, tensor<1xbf16>) -> tensor<1x16x16x2xbf16>
    return %2 : tensor<1x16x16x2xbf16>
  }
}

