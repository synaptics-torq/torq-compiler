// TORQ_FP_MAX_TOL: 1
module attributes {tf_saved_model.semantics} {
  func.func @main(%arg0: tensor<1x16x16x2xbf16> {ml_program.identifier = "serving_default_input_0:0", tf_saved_model.index_path = ["input_0"]}) -> (tensor<1x16x16x2xbf16> {ml_program.identifier = "StatefulPartitionedCall_1:0", tf_saved_model.index_path = ["output_0"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = "tosa.const"() <{value = dense<[11.0,22.0]> : tensor<2xbf16>}> : () -> tensor<2xbf16>
    %1 = "tosa.const"() <{value = dense<[
      [[[15.0],[7.0]], [[18.0],[8.0]], [[12.0],[8.5]], [[15.0],[18.0]], [[17.0],[12.5]]],
      [[[14.0],[4.0]], [[9.0], [9.0]], [[10.0],[9.0]], [[2.0],[4.5]], [[8.0],[6.0]]],
      [[[16.0],[9.5]], [[11.0],[6.0]], [[19.2],[7.0]], [[16.0],[6.0]], [[1.0],[3.5]]],
      [[[15.0],[7.0]], [[18.0],[8.0]], [[12.0],[8.5]], [[5.0],[18.0]], [[17.0],[1.5]]],
      [[[15.0],[17.0]],[[18.0],[8.0]], [[12.0],[18.5]],[[10.0],[18.0]], [[17.0],[12.5]]]]> : tensor<5x5x2x1xbf16>}> : () -> tensor<5x5x2x1xbf16>
    %2 = tosa.depthwise_conv2d %arg0, %1, %0 {
      dilation = array<i64: 1, 1>,
      pad = array<i64: 2, 2, 2, 2>,
      stride = array<i64: 1, 1>
    } : (tensor<1x16x16x2xbf16>, tensor<5x5x2x1xbf16>, tensor<2xbf16>) -> tensor<1x16x16x2xbf16>
    return %2 : tensor<1x16x16x2xbf16>
  }
}
