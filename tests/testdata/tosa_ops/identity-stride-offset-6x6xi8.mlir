module {
  func.func @main(%arg0: tensor<1x8x8x1xi8> {ml_program.identifier = "serving_default_input_0:0", tf_saved_model.index_path = ["input_0"]}) -> (tensor<1x6x6x1xi8> {ml_program.identifier = "StatefulPartitionedCall_1:0", tf_saved_model.index_path = ["output_0"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = tosa.identity %arg0 : (tensor<1x8x8x1xi8>) -> tensor<1x8x8x1xi8>
    %extracted_slice = tensor.extract_slice %0[0, 1, 1, 0] [1, 6, 6, 1] [1, 1, 1, 1] : tensor<1x8x8x1xi8> to tensor<1x6x6x1xi8>
    return %extracted_slice : tensor<1x6x6x1xi8>
  }
}

