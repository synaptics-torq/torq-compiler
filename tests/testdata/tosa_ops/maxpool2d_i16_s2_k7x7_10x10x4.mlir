module {
  func.func @main(%arg0: tensor<1x10x10x4xi16> {ml_program.identifier = "serving_default_input_0:0", tf_saved_model.index_path = ["input_0"]}) -> (tensor<1x5x5x4xi16> {ml_program.identifier = "PartitionedCall_1:0", tf_saved_model.index_path = ["output_0"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 7, 7>, pad = array<i64: 2, 3, 2, 3>, stride = array<i64: 2, 2>} : (tensor<1x10x10x4xi16>) -> tensor<1x5x5x4xi16>
    return %0 : tensor<1x5x5x4xi16>
  }
}

