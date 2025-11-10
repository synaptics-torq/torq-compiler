module {
  func.func @main(%500: tensor<1x20x20x256xi8>) -> (tensor<1x40x40x256xi8> {ml_program.identifier = "ResizeOp", tf_saved_model.index_path = ["output"]}) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %501 = tosa.resize %500 {border = array<i64: 3, 3>, mode = "NEAREST_NEIGHBOR", offset = array<i64: 1, 1>, scale = array<i64: 4, 2, 4, 2>} : (tensor<1x20x20x256xi8>) -> tensor<1x40x40x256xi8>
    return %501 : tensor<1x40x40x256xi8>
  }
}