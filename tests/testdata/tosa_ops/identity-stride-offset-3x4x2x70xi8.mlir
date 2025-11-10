module {
  func.func @main(%arg0: tensor<6x5x2x77xi8> {ml_program.identifier = "serving_default_input_0:0", tf_saved_model.index_path = ["input_0"]}) -> (tensor<1x3x4x2x70xi8> {ml_program.identifier = "StatefulPartitionedCall_1:0", tf_saved_model.index_path = ["output_0"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %3 = tosa.identity %arg0 : (tensor<6x5x2x77xi8> ) -> tensor<6x5x2x77xi8> 
    %extracted_slice = tensor.extract_slice %3[0, 1, 1, 1] [3, 4, 2, 70] [1, 1, 1, 1] : tensor<6x5x2x77xi8> to tensor<3x4x2x70xi8>
    %4 = tosa.reshape %extracted_slice {new_shape = array<i64: 1, 3, 4, 2, 70>} : (tensor<3x4x2x70xi8>) -> tensor<1x3x4x2x70xi8>
    return %4 : tensor<1x3x4x2x70xi8>
  }
}

