module {
  func.func @main(%arg0: tensor<1x56x56x256xi8>) -> (tensor<1x28x28x256xi8> {ml_program.identifier = "StatefulPartitionedCall_1:0", tf_saved_model.index_path = ["output_0"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = tosa.max_pool2d %arg0 {kernel = array<i64: 2, 2>, pad = array<i64: 0, 0, 0, 0>, stride = array<i64: 2, 2>} : (tensor<1x56x56x256xi8>) -> tensor<1x28x28x256xi8>
    return %0 : tensor<1x28x28x256xi8>
  }
}

