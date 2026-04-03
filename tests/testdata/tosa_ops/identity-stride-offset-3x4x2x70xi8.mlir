module {
  func.func @main(%arg0: tensor<6x5x2x77xi8> {ml_program.identifier = "serving_default_input_0:0", tf_saved_model.index_path = ["input_0"]}) -> (tensor<1x3x4x2x70xi8> {ml_program.identifier = "StatefulPartitionedCall_1:0", tf_saved_model.index_path = ["output_0"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = tosa.identity %arg0 : (tensor<6x5x2x77xi8>) -> tensor<6x5x2x77xi8>
    %1 = tosa.const_shape  {values = dense<[0, 1, 0, 1]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %2 = tosa.const_shape  {values = dense<[3, 4, 2, 70]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %3 = tosa.slice %0, %1, %2 : (tensor<6x5x2x77xi8>, !tosa.shape<4>, !tosa.shape<4>) -> tensor<3x4x2x70xi8>
    %4 = tosa.const_shape  {values = dense<[1, 3, 4, 2, 70]> : tensor<5xindex>} : () -> !tosa.shape<5>
    %5 = tosa.reshape %3, %4 : (tensor<3x4x2x70xi8>, !tosa.shape<5>) -> tensor<1x3x4x2x70xi8>
    return %5 : tensor<1x3x4x2x70xi8>
  }
}

