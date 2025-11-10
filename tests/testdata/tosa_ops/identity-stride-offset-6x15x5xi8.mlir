module {
  func.func @main(%arg0: tensor<1x10x80x90xi8> {ml_program.identifier = "serving_default_input_0:0", tf_saved_model.index_path = ["input_0"]}) -> (tensor<1x6x15x5xi8> {ml_program.identifier = "StatefulPartitionedCall_1:0", tf_saved_model.index_path = ["output_0"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %3 = tosa.identity %arg0 : (tensor<1x10x80x90xi8> ) -> tensor<1x10x80x90xi8> 
    %extracted_slice = tensor.extract_slice %3[0, 0, 1, 1] [1, 6,15, 5] [1, 1, 1, 1] : tensor<1x10x80x90xi8> to tensor<6x15x5xi8>
    %4 = tosa.reshape %extracted_slice {new_shape = array<i64: 1, 6, 15, 5>} : (tensor<6x15x5xi8>) -> tensor<1x6x15x5xi8>
    return %4 : tensor<1x6x15x5xi8>
  }
}

