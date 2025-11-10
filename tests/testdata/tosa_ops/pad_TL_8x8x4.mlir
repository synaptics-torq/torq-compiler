module {
  func.func @main(%input: tensor<1x8x8x4xi8>) -> (tensor<1x9x9x4xi8>) attributes {tf_saved_model.exported_names = ["serving_default"]} {
    %zero = arith.constant 0 : i8
    %padded = tensor.pad %input low[0, 1, 1, 0] high[0, 0, 0, 0] {
      ^bb0(%i0: index, %i1: index, %i2: index, %i3: index):
        tensor.yield %zero : i8
    } : tensor<1x8x8x4xi8> to tensor<1x9x9x4xi8>
    return %padded : tensor<1x9x9x4xi8>
  }
}


