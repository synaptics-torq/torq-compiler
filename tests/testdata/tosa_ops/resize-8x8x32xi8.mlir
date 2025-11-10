module {
  func.func @main(%arg0: tensor<1x8x8x32xi8> {ml_program.identifier = "serving_default_input_1:0", tf_saved_model.index_path = ["input_0"]}) -> (tensor<1x16x16x32xi8> {ml_program.identifier = "ResizeOp", tf_saved_model.index_path = ["output"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %0 = tosa.resize %arg0 {mode = "NEAREST_NEIGHBOR", scale = array<i64: 4, 2, 4, 2>, offset = array<i64: -1,-1>, border = array<i64: 1, 1>} : (tensor<1x8x8x32xi8>) -> tensor<1x16x16x32xi8>
    return %0 : tensor<1x16x16x32xi8>
  }
}
