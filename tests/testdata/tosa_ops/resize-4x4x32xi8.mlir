module {
  func.func @main(%arg0: tensor<1x4x4x32xi8> {ml_program.identifier = "serving_default_input_1:0", tf_saved_model.index_path = ["input_0"]}) -> (tensor<1x8x8x32xi8> {ml_program.identifier = "ResizeOp", tf_saved_model.index_path = ["output"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = tosa.const_shape  {values = dense<[4, 2, 4, 2]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %1 = tosa.const_shape  {values = dense<-1> : tensor<2xindex>} : () -> !tosa.shape<2>
    %2 = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
    %3 = tosa.resize %arg0, %0, %1, %2 {mode = NEAREST_NEIGHBOR} : (tensor<1x4x4x32xi8>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x8x8x32xi8>
    return %3 : tensor<1x8x8x32xi8>
  }
}

