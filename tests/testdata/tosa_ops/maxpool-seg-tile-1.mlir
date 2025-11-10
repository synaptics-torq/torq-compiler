module {
  func.func @main(%58: tensor<1x28x28x512xi8>) -> (tensor<1x14x14x512xi8> {ml_program.identifier = "StatefulPartitionedCall_1:0", tf_saved_model.index_path = ["output_0"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %59 = tosa.max_pool2d %58 {kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x28x28x512xi8>) -> tensor<1x14x14x512xi8>
    return %59 : tensor<1x14x14x512xi8>
  }
}

