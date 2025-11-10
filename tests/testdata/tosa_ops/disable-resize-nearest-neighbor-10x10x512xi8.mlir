module {
  func.func @main(%466: tensor<1x10x10x512xi8>) -> (tensor<1x20x20x512xi8> {ml_program.identifier = "ResizeOp", tf_saved_model.index_path = ["output"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %467 = tosa.resize %466 {border = array<i64: 3, 3>, mode = "NEAREST_NEIGHBOR", offset = array<i64: 1, 1>, scale = array<i64: 4, 2, 4, 2>} : (tensor<1x10x10x512xi8>) -> tensor<1x20x20x512xi8>
    return %467 : tensor<1x20x20x512xi8>
  }
}