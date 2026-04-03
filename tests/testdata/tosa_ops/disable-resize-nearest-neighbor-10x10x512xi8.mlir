module {
  func.func @main(%arg0: tensor<1x10x10x512xi8>) -> (tensor<1x20x20x512xi8> {ml_program.identifier = "ResizeOp", tf_saved_model.index_path = ["output"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = tosa.const_shape  {values = dense<[4, 2, 4, 2]> : tensor<4xindex>} : () -> !tosa.shape<4>
    %1 = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
    %2 = tosa.const_shape  {values = dense<3> : tensor<2xindex>} : () -> !tosa.shape<2>
    %3 = tosa.resize %arg0, %0, %1, %2 {mode = NEAREST_NEIGHBOR} : (tensor<1x10x10x512xi8>, !tosa.shape<4>, !tosa.shape<2>, !tosa.shape<2>) -> tensor<1x20x20x512xi8>
    return %3 : tensor<1x20x20x512xi8>
  }
}

