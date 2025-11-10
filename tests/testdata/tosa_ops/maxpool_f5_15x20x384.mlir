module {
  func.func @main(%arg0: tensor<1x15x20x384xi8> {ml_program.identifier = "serving_default_input_0:0", tf_saved_model.index_path = ["input_0"]}) -> (tensor<1x15x20x384xi8> {ml_program.identifier = "PartitionedCall_1:0", tf_saved_model.index_path = ["output_0"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 5, 5>, pad = array<i64: 2, 2, 2, 2>, stride = array<i64: 1, 1>} : (tensor<1x15x20x384xi8>) -> tensor<1x15x20x384xi8>
    return %0 : tensor<1x15x20x384xi8>
  }
}

