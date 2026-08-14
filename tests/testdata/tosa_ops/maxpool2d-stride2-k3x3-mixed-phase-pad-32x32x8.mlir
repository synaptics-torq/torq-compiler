module {
  func.func @main(%arg0: tensor<1x32x32x8xi8> {ml_program.identifier = "serving_default_keras_tensor:0", tf_saved_model.index_path = ["keras_tensor"]}) -> (tensor<1x16x16x8xi8> {ml_program.identifier = "PartitionedCall_1:0", tf_saved_model.index_path = ["output_0"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 3, 3>, pad = array<i64: 0, 1, 1, 0>, stride = array<i64: 2, 2>} : (tensor<1x32x32x8xi8>) -> tensor<1x16x16x8xi8>
    return %0 : tensor<1x16x16x8xi8>
  }
}
