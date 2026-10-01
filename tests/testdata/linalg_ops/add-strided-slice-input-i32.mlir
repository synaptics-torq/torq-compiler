// Regression: a two-tensor add whose inputs have different strides. FoldConvert hands the
// add the column slice of the matmul result as an LRAM subview, next to a dense operand.
module {
  func.func @main(%lhs: tensor<32x64xi8>, %rhs: tensor<64x48xi8>, %other: tensor<32x40xi32>) -> tensor<32x40xi32> {
    %c0 = arith.constant 0 : i32
    %e = tensor.empty() : tensor<32x48xi32>
    %z = linalg.fill ins(%c0 : i32) outs(%e : tensor<32x48xi32>) -> tensor<32x48xi32>
    %mm = linalg.matmul ins(%lhs, %rhs : tensor<32x64xi8>, tensor<64x48xi8>) outs(%z : tensor<32x48xi32>) -> tensor<32x48xi32>
    %s = tensor.extract_slice %mm[0, 0] [32, 40] [1, 1] : tensor<32x48xi32> to tensor<32x40xi32>
    %e2 = tensor.empty() : tensor<32x40xi32>
    %r = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
    } ins(%s, %other : tensor<32x40xi32>, tensor<32x40xi32>) outs(%e2 : tensor<32x40xi32>) {
    ^bb0(%a: i32, %b: i32, %out: i32):
      %m = arith.addi %a, %b : i32
      linalg.yield %m : i32
    } -> tensor<32x40xi32>
    return %r : tensor<32x40xi32>
  }
}
